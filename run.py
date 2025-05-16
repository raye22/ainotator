"""Annotate CMC utterances with AI

Automates utterance-level annotation for computer-mediated communication (CMC)
data using either OpenAI chat models or a local Llama-3.1 model.


1. Accepted workbook layouts

- Yusra style
    Msg# | Utterance # | Date | Time | User ID | Gender | Message
    (no Reply to_ID column)

Soyeon style
    Msg# | Date | Category | User ID | Reply to_ID | Message
    (may contain several “Original post” rows per thread)

The loader `_load_xlsx()` normalises both layouts to a canonical dataframe with
these columns:

    Msg# | Utterance # | Date | Time | User ID | Gender | Message | Reply to_ID

Missing optional fields are filled with empty strings.

2. Prompt construction

- Background
         - Yusra : the “JuvieThrowaw” juvenile-detention narrative
         - Soyeon: the r/gaming “progressed farther in Terraria” scenario

- Thread starters
         All rows where Msg# == 1 (Yusra) or Category == "Original post"
         (Soyeon); included in every prompt to anchor the conversation.

- Local context
         Soyeon (reply-aware):
             PREV = most recent earlier message whose User ID equals Reply to_ID
         Yusra (fallback):
             PREV = previous row
             NEXT = next row

- User metadata line (UserID, Gender, Time, Utterance #)

3. Reasoning

With the --cot flag the prompt adds:
    Think step-by-step inside [REASON]…[/REASON] before the answer.
For the o3 model an extra sentence asks it to copy any hidden reasoning into the
same bracketed block.

4. Expected model output
Exactly one JSON object wrapped in [ANNOT]…[/ANNOT]:

    [ANNOT]{"act":"<ACT>","politeness":"<POL>","meta":"<META>"}[/ANNOT]

Fields
    act: one label from the CMC communicative-act taxonomy
    politeness: Herring (+/-P, +/-N) or Culpeper codes
    meta: optional meta-acts: non-bona fide, reported

5. Run-time features
- Resumable: rows with a non-empty act cell are skipped on rerun.
- Moderation: rows rejected by OpenAI policy are marked __FLAGGED__.
- Checkpoints: workbook is saved every 20 rows.
- Audit trail: raw prompts/responses and cleaned annotations are streamed to
  annotations/<model_tag>_seed_<seed>[_cot]/.

6. Note:
    We only support Llama-3.1 and OpenAI models.
    - o3-2025-04-16
    - gpt-4o-2024-08-06
    - meta-llama/Llama-3.1-8B-Instruct
"""

__author__ = "The AInotator authors"
__version__ = "0.1.0"
__license__ = "MIT"


import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import openai
import pandas as pd
from tqdm import tqdm

FIXED_SEEDS = [93187, 95617, 98473, 101089, 103387]

ALLOWED_ACTS: List[str] = [
    "Accept",
    "Apologize",
    "Behave",
    "Claim",
    "Congratulate",
    "Desire",
    "Direct",
    "Elaborate",
    "Greet",
    "Inform",
    "Inquire",
    "Invite",
    "Manage",
    "React",
    "Reject",
    "Repair",
    "Request",
    "Thank",
]
ALLOWED_POLITENESS: List[str] = ["+P", "+N", "-P", "-N"]
ALLOWED_META: List[str] = ["non-bona fide", "reported"]

START_TAG = "[ANNOT]"
END_TAG = "[/ANNOT]"
REASON_START = "[REASON]"
REASON_END = "[/REASON]"


BACKGROUND_YUSRA = (
    "Background: A Reddit user (“JuvieThrowaw”) shares that as a teenager they "
    "fatally shot their mother's abusive boyfriend after he harmed their sister, "
    "served juvenile time, and now struggles with whether to disclose this past "
    "to new friends and partners."
)

BACKGROUND_SOYEON = (
    "Background: A Reddit user (CallSign_Fjor) posts in r/gaming that their "
    "friends lose interest in finishing Terraria and other games whenever the "
    "user progresses farther, even though the user avoids using end-game gear "
    "and shares resources. The post asks why some players feel 'invalidated' "
    "by someone else’s faster progression."
)


# load either Yusra-style or Soyeon-style workbook and
# return a DataFrame with canonical column names expected downstream
def _load_xlsx(path: str) -> pd.DataFrame:
    """Load either the Yusra sheet or the Soyeon sheet and return a
    canonical DataFrame with columns

        Msg# | Utterance # | Date | Time | User ID | Gender | Message | Reply to_ID

    - Add missing optional cols (Gender, Time, Reply to_ID) as empty strings.
    - Create Utterance # when absent (running count inside each Msg# thread).
    """

    df = pd.read_excel(path, engine="openpyxl")

    # ensure Message col exists
    if "Message" not in df.columns:
        raise ValueError("Workbook must contain a 'Message' column")

    # add missing optional columns
    for col in ("Gender", "Time", "Reply to_ID"):
        if col not in df.columns:
            df[col] = ""

    # create Utterance # if absent (Yusra has it; Soyeon doesn't)
    if "Utterance #" not in df.columns:
        df["Utterance #"] = (
            df.groupby("Msg#", sort=False).cumcount() + 1
            if "Msg#" in df.columns
            else range(1, len(df) + 1)
        )

    # sanity-check required cols
    required = {"Msg#", "User ID", "Message"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Workbook missing required column(s): {missing}")

    # tidy order
    base_cols = [
        "Msg#",
        "Utterance #",
        "Date",
        "Time",
        "User ID",
        "Gender",
        "Message",
        "Reply to_ID",
    ]
    df = df[
        [c for c in base_cols if c in df.columns]
        + [c for c in df.columns if c not in base_cols]
    ]

    return df


def _get_local_context(idx: int, df: pd.DataFrame) -> Tuple[str, str]:
    """
    Return (prev_msg, next_msg) for row idx.

    - If the sheet has a non-empty 'Reply to_ID' column we fetch only the
      *parent* message (prev_msg) and leave next_msg blank, because real-world
      reply trees rarely need a “next child” for local context.
    - Otherwise fall back to simple previous / next rows (Yusra layout).
    """
    row = df.iloc[idx]

    if "Reply to_ID" in df.columns and df["Reply to_ID"].astype(str).str.strip().any():
        prev_msg = ""
        reply_to = str(row.get("Reply to_ID", "")).strip()
        if reply_to:
            earlier = df.iloc[:idx]
            mask = earlier["User ID"].astype(str) == reply_to
            if mask.any():
                prev_msg = earlier.loc[mask, "Message"].iloc[-1]
        return prev_msg, ""  # <-- next_msg intentionally blank

    # sequential fallback (Yusra)
    prev_msg = df.at[idx - 1, "Message"] if idx > 0 else ""
    next_msg = df.at[idx + 1, "Message"] if idx < len(df) - 1 else ""
    return prev_msg, next_msg


def _build_messages(
    system_prompt: str,
    context: Tuple[str, str, str],
    include_cot: bool,
    global_context: str,
    user_meta: str,
    model_tag: str,  # "gpt-4o", "o3", or "llama"
) -> List[Dict]:
    """Compose the list of chat messages for OpenAI / vLLM, including
    global context and user metadata."""

    prev_msg, target_msg, next_msg = context

    # user-side blocks
    user_block = (
        f"{user_meta}"
        f"[PREV] {prev_msg or '<NONE>'}\n"
        f"[TARGET] {target_msg}\n"
        f"[NEXT] {next_msg or '<NONE>'}\n"
    )

    reasoning_block = ""
    if include_cot:
        reasoning_block = (
            "\nThink step-by-step inside [REASON]…[/REASON] before the answer."
        )
        if model_tag == "o3":
            reasoning_block += (
                "\nIf you used hidden or internal reasoning anywhere, copy **all** of that "
                "reasoning verbatim inside the same [REASON]…[/REASON] block."
            )

    format_block = (
        "\nReturn the annotation as one JSON object wrapped EXACTLY like:\n"
        f'{START_TAG}{{"act":"<ACT>","politeness":"<POL>","meta":"<META>"}}{END_TAG}'
    )

    # system block
    system_block = f"{global_context.strip()}\n\n{system_prompt}"

    return [
        {"role": "system", "content": system_block},
        {"role": "user", "content": user_block + reasoning_block + format_block},
    ]


def _parse_annotation(text: str) -> Dict:
    """Extract reasoning (optional) and the JSON annotation; validate all fields."""

    # REASON block (optional)
    reason = ""
    if REASON_START in text and REASON_END in text:
        reason = text.split(REASON_START, 1)[1].split(REASON_END, 1)[0].strip()

    # ANNOT block (required)
    if START_TAG not in text or END_TAG not in text:
        raise ValueError("wrapper tags not found")

    json_str = text.split(START_TAG, 1)[1].split(END_TAG, 1)[0].strip()
    anno = json.loads(json_str)

    # 1) communicative act
    act = str(anno.get("act", "")).strip()
    if act not in ALLOWED_ACTS:
        raise ValueError(f"invalid act: {act}")

    # 2) politeness (+ optional subtype)
    raw_pol = str(anno.get("politeness", "") or "")
    raw_pol = raw_pol.replace("–", "-").replace("—", "-").strip()

    meta_field = str(anno.get("meta", "") or "").strip()  # safe default ""

    if raw_pol.lower() == "none":
        pol, meta_from_pol = "", ""
    else:
        if "[" in raw_pol and "]" in raw_pol:
            base, subtype = raw_pol.split("[", 1)
            pol = base.strip()
            meta_from_pol = subtype.rstrip("]").strip()
        else:
            pol = raw_pol
            meta_from_pol = ""

    if pol and pol not in ALLOWED_POLITENESS:
        raise ValueError(f"invalid politeness: {pol}")

    # 3) meta tags
    combined_meta = ", ".join(filter(None, [meta_field, meta_from_pol]))
    clean_meta: List[str] = []
    for tag in [t.strip() for t in combined_meta.split(",") if t.strip()]:
        if tag not in ALLOWED_META:
            logging.warning(f"unrecognized meta tag: {tag}")  # keep but warn
        else:
            clean_meta.append(tag)
    meta = ", ".join(clean_meta)

    return {"act": act, "politeness": pol, "meta": meta, "reason": reason}


def _annotate_row(
    row_idx: int,
    df: pd.DataFrame,
    sys_prompt: str,
    model: str,
    max_tries: int,
    include_cot: bool,
    global_context: str,
    user_meta: str,
    llm=None,
    tokenizer=None,
) -> Tuple[Dict, List[Dict]]:
    """Annotate one DataFrame row, retrying with incremental seeds.
    Works with either OpenAI chat models (gpt-4o, o3) or a local vLLM
    instance of Llama-3.1-8B-Instruct.
    """
    raw_records: List[Dict] = []

    prev_msg, next_msg = _get_local_context(row_idx, df)
    targ_msg = df.at[row_idx, "Message"]

    base_seed = FIXED_SEEDS[0]

    for attempt in range(max_tries):
        seed = base_seed + attempt
        messages = _build_messages(
            sys_prompt,
            (prev_msg, targ_msg, next_msg),
            include_cot,
            global_context,
            user_meta,
            (
                "o3"
                if model.startswith("o3")
                else "gpt-4o" if model.startswith("gpt-4o") else "llama"
            ),
        )

        # inference
        if llm is None:  # OpenAI endpoint
            try:
                if model.startswith("o3"):
                    resp = openai.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=1.0,
                        top_p=1.0,
                        seed=seed,
                    )
                else:
                    resp = openai.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=0.7,
                        top_p=0.95,
                        seed=seed,
                    )
            except openai.BadRequestError as e:
                # flagged by moderation — mark and bail out
                logging.warning(
                    f"Row {row_idx} seed {seed} flagged by policy; skipping."
                )
                raise RuntimeError("policy_flagged")
            content = resp.choices[0].message.content
            ts = resp.created
        else:  # local Llama-3.1 via vLLM
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            out = llm.generate(
                [prompt],
                sampling_params=SamplingParams(
                    temperature=0.7,
                    top_p=0.95,
                    seed=seed,
                ),
            )[0].outputs[0]
            content = out.text
            ts = time.time()

        raw_records.append(
            {
                "row_idx": row_idx,
                "seed": seed,
                "prompt": json.dumps(messages, ensure_ascii=False),
                "response": content,
                "timestamp": ts,
            }
        )

        try:
            anno = _parse_annotation(content)
            # make sure reasoning was really captured
            if include_cot and len(anno.get("reason", "")) < 5:
                raise ValueError("reasoning too short – likely not captured")

            anno.update({"row_idx": row_idx, "seed": seed})
            return anno, raw_records

        except (json.JSONDecodeError, ValueError) as e:
            logging.warning(f"Row {row_idx} seed {seed} parse error: {e}")
            time.sleep(2**attempt)

    raise RuntimeError(f"row {row_idx}: parse failed after {max_tries} tries")


def main() -> None:
    """Entry point: load data, replay prior annotations, annotate rows, checkpoint,
    and save final XLSX."""
    CHECKPOINT_EVERY = 20  # rows after which we write df back to Excel

    # 1) CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", default="data/Yusra.xlsx")
    parser.add_argument("--model", default="gpt-4o-2024-08-06")
    parser.add_argument("--max_tries", type=int, default=20)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cot", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s", level=logging.INFO
    )
    logging.info("Starting annotation run")

    # load & normalise workbook (handles either layout)
    df = _load_xlsx(args.xlsx)

    # replay any previously‐logged annotations so we skip them on resume
    if args.model.startswith("gpt-4o"):
        model_tag = "gpt-4o"
    elif args.model.startswith("o3"):
        model_tag = "o3"
    elif "llama" in args.model.lower():
        model_tag = "llama"
    else:
        raise ValueError("Unknown model. Only support Llama-3.1 and OpenAI API.")

    primary_seed = FIXED_SEEDS[0]
    cot_suffix = "_cot" if args.cot else ""
    root = (
        Path("soyeon_annotations")
        if "Category" in df.columns
        else Path("yusra_annotations")
    )
    out_dir = root / f"{model_tag}_seed_{primary_seed}{cot_suffix}"
    clean_path = out_dir / "annot_clean.csv"

    if clean_path.exists():
        prev = pd.read_csv(clean_path)
        for _, r in prev.iterrows():
            ridx = int(r["row_idx"])
            act = r["act"]
            if act and act not in ("__FAILED__", "__FLAGGED__"):
                df.at[ridx, "act"] = act
    # 3) build dynamic global context
    first_posts = df.loc[df["Msg#"] == 1, "Message"].dropna().tolist()
    thread_summary = BACKGROUND_YUSRA
    if "Category" in df.columns:  # Soyeon layout
        thread_summary = BACKGROUND_SOYEON
    dynamic_global_context = "\n\n".join(
        [
            thread_summary,
            "Thread starter messages:\n" + "\n".join(f"- {m}" for m in first_posts),
        ]
    )

    # 4) local / remote setup
    is_llama = model_tag == "llama"
    tokenizer = llm = None
    if is_llama:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        llm = LLM(model=args.model, dtype="bfloat16")

    # 5) ensure annotation columns exist
    for col in ("act", "politeness", "meta"):
        if col not in df.columns:
            df[col] = ""

    # 6) determine rows to annotate
    todo_idx = df.index[df["act"].fillna("") == ""]
    if args.debug:
        todo_idx = todo_idx[:10]
        logging.info("Debug mode: first 10 only")

    system_prompt = Path("system_prompt.md").read_text(encoding="utf-8")
    pbar = tqdm(todo_idx, desc="Annotating", unit="row")

    # 7) main loop
    for count, idx in enumerate(pbar, start=1):
        row = df.iloc[idx]
        user_meta = (
            f"UserID: {row['User ID']}, "
            f"Gender: {row['Gender']}, "
            f"Time: {row['Time']}, "
            f"Utterance#: {row['Utterance #']}\n"
        )

        try:
            anno, raws = _annotate_row(
                idx,
                df,
                system_prompt,
                args.model,
                args.max_tries,
                include_cot=args.cot,
                global_context=dynamic_global_context,
                user_meta=user_meta,
                llm=llm,
                tokenizer=tokenizer,
            )

            # write back to DataFrame
            df.loc[idx, ["act", "politeness", "meta"]] = [
                anno["act"],
                anno["politeness"],
                anno["meta"],
            ]

            # append logs
            pd.DataFrame([anno]).to_csv(
                out_dir / "annot_clean.csv",
                mode="a",
                header=not clean_path.exists(),
                index=False,
            )
            pd.DataFrame(raws).to_csv(
                out_dir / "annot_raw.csv",
                mode="a",
                header=not (out_dir / "annot_raw.csv").exists(),
                index=False,
            )
            df.iloc[[idx]].to_csv(
                out_dir / "annot_seq.csv",
                mode="a",
                header=not (out_dir / "annot_seq.csv").exists(),
                index=False,
            )

            logging.info(f"Annotated row {idx}")

        except RuntimeError as exc:
            flag = "__FLAGGED__" if str(exc) == "policy_flagged" else "__FAILED__"
            logging.error(exc)
            df.loc[idx, ["act", "politeness", "meta"]] = [flag, "", ""]
            df.iloc[[idx]].to_csv(
                out_dir / "annot_seq.csv",
                mode="a",
                header=not (out_dir / "annot_seq.csv").exists(),
                index=False,
            )

        # periodic checkpoint to XLSX
        if not args.debug and count % CHECKPOINT_EVERY == 0:
            df.to_excel(args.xlsx, index=False)
            logging.info("Checkpoint saved to Excel")

    # 8) final save
    if not args.debug:
        df.to_excel(args.xlsx, index=False)
        logging.info("Annotation run complete – final workbook saved")
    else:
        logging.info("Debug run complete — no changes written to workbook")


if __name__ == "__main__":
    main()
