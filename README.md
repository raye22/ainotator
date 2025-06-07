# 🤖 AInotator: Annotate Computer-Mediated Discourse with LLMs

Can LLMs understand what is meant, not just what is said?
**AInotator** puts this question to the test by automatically annotating computer-mediated discourse using state-of-the-art language models.

For each utterance in context, we assign:

- **Communicative Act Labels** (e.g., Accept, Request, Reject)
- **Politeness Tags** following Brown & Levinson (1987), Herring (1994) and Culpeper (2011)
- **Meta-Acts** (e.g., [non-bona fide], [reported])

Annotations are generated through structured prompting and saved in a reproducible, debuggable format. The tool supports:

- **Multiple model backends**: OpenAI (GPT-4o, O3) and local Llama-3.1
- **Corpus-agnostic processing**: Works with any CMC dataset structure
- **Resumable runs** with comprehensive progress logging
- **Chain-of-thought (CoT) reasoning** mode for enhanced transparency
- **Robust error handling** with automatic retry and reasoning validation
- **Reproducibility** through fixed seeds and complete audit trails

## 🧭 Annotation Schema

The model follows the **CMC Act Taxonomy** (Herring, Das, and Penumarthy 2005; revised 2024) adapted from CMC pragmatics and politeness theory.

### 🎙️ Communicative Act Labels (18 total)

| Label          | Definition | Example                                                              |
|----------------|------------|----------------------------------------------------------------------|
| **Accept**     | Concur, agree, acquiesce, approve; acknowledge | "Exactly this."; "I agree" |
| **Apologize**  | Humble oneself, self-deprecate | "Sorry this happened to your family." |
| **Behave**     | Perform a virtual action | "*dances with joy"; "*sips tea" |
| **Claim**      | Make subjective assertion; unverifiable in principle | "I do not understand the mentality of people who..." |
| **Congratulate** | Celebrate/praise accomplishment; encourage; validate | "Well done!"; "You've got this!" |
| **Desire**     | Want, hope, wish; promise, predict; hypothetical | "I wish they'd just play the game together." |
| **Direct**     | Command, demand; prohibit; permit; advise | "You should try something else." |
| **Elaborate**  | Explain or paraphrase previous utterance | "This isn't the first time it happened..." |
| **Greet**      | Greeting, leave-taking; formulaic well-being inquiries | "Hello"; "How are you?" |
| **Inform**     | Provide "factual" information (verifiable in principle) | "I recently played Terraria with friends..." |
| **Inquire**    | Seek information; make neutral proposals | "What's up with people being upset about this?" |
| **Invite**     | Seek participation; suggest; offer | "You might want to post this in another subreddit." |
| **Manage**     | Organize, prompt, focus, open/close discussions | "I have two thoughts about that..." |
| **React**      | Show listenership, engagement | "Lmao this is so dramatic."; "wow" |
| **Reject**     | Disagree, dispute, challenge | "Dude! You came here for answers and you are NOT listening." |
| **Repair**     | Clarify or seek clarification; correct misunderstanding | "Did you mean 'school holiday'?" |
| **Request**    | Seek action politely | "Can someone explain this to me?" |
| **Thank**      | Express gratitude, appreciation | "Thanks for saying this." |

### 🪞 Politeness & Impoliteness

Based on **Brown & Levinson (1987)** Politeness Theory: Positive politeness aims to enhance the addressee's self-esteem, while negative politeness respects their autonomy.

| Code      | Meaning                                     | Examples                                        |
|-----------|---------------------------------------------|-------------------------------------------------|
| **+P**    | Affirm positive face (desire to be liked, appreciated) | Compliments, support, friendly humor |
| **+N**    | Respect negative face (desire for autonomy) | Hedging, deference, giving options |
| **-P**    | Attack positive face | Insults, mocking, condescension |
| **-N**    | Attack negative face | Commands, intrusive questions, impositions |

**Impoliteness subtypes** (Culpeper 2011): `[Insult]`, `[Condescension]`, `[Dismissal]`, `[Silencer]`, `[Threat]`, `[Negative association]`

### 🏷️ Meta-Acts

| Tag            | Description                                         |
|----------------|-----------------------------------------------------|
| **[non-bona fide]** | Sarcasm, irony, jokes, rhetorical questions     |
| **[reported]**      | Quoting or paraphrasing others' speech/thoughts |

---

## 🚀 Usage

### Basic Commands
```bash
# Annotate with default settings (GPT-4o)
python annotate.py --xlsx your_data.xlsx

# Enable chain-of-thought reasoning
python annotate.py --xlsx your_data.xlsx --cot

# Use O3 model with reasoning
python annotate.py --xlsx your_data.xlsx --model o3-2025-04-16 --cot

# Debug mode (first 10 rows only)
python annotate.py --xlsx your_data.xlsx --debug
```

### Supported Models
- **OpenAI**: `gpt-4o-2024-08-06`, `o3-2025-04-16`
- **Local**: `meta-llama/Llama-3.1-8B-Instruct` (via vLLM)

### Input Data Requirements
Your Excel file should contain at minimum:
- `Msg#`: Message thread identifier
- `User ID`: Speaker identifier  
- `Message`: The utterance text

Optional columns (automatically handled):
- `Utterance #`: Position in thread
- `Gender`, `Time`: User metadata
- `Reply to_ID`: For threaded conversations

## 📂 Output Structure

Results are organized in `{corpus}_annotations/{model}_seed_{seed}[_cot]/`:

```
yusra_annotations/gpt-4o_seed_93187_cot/
├── annot_raw.csv           # All model attempts (including failures)
├── annot_clean.csv         # Successfully parsed annotations
├── successful_inputs.csv   # Clean input/output pairs for analysis
└── annot_seq.csv          # Sequential processing log
```

### Output Fields
- **act**: Primary communicative act (required)
- **politeness**: Politeness code with optional subtype (e.g., "-P [Insult]")
- **meta**: Meta-act tags (comma-separated if multiple)
- **reason**: Step-by-step reasoning (when --cot enabled)

## 🔧 Key Features

### Robust Processing
- **Automatic retry logic** with exponential backoff
- **Enhanced reasoning validation** for O3 models
- **Dash symbol normalization** (handles –, —, −, etc.)
- **Content policy handling** for sensitive content

### Corpus Flexibility
- **Dynamic context building** adapts to threaded vs. sequential conversations
- **Missing metadata handling** works with incomplete user information
- **Universal system prompt** works across different CMC datasets

### Quality Assurance
- **Fixed seed reproducibility** with comprehensive audit trails
- **Checkpoint saving** every 20 rows for long runs
- **Progress tracking** with success/failure/flagged counts
- **Resumable processing** skips already-annotated rows

## 💡 Why It Matters

Manual annotation of online discourse is slow, inconsistent, and hard to scale. Traditional rule-based systems struggle with context, sarcasm, and pragmatic nuance.

AInotator offers a **practical, theory-aware solution** that:
- Captures communicative intent beyond surface form
- Handles non-literal language (sarcasm, irony, rhetorical questions)
- Maintains theoretical grounding in established CMC frameworks
- Scales to large datasets while preserving annotation quality

Perfect for CMC researchers studying **stance, identity, politeness, conflict, and solidarity** at scale.

> LLMs may be changing the game — but we still define the rules.

## 🛠️ Installation & Setup

```bash
# Clone repository
git clone https://github.com/Wang-Haining/ainotator.git
cd ainotator

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key (if using OpenAI models)
export OPENAI_API_KEY="your-api-key-here"

# For local Llama models, ensure vLLM is properly configured
```

## 📊 Performance Notes

- **GPT-4o**: Fast, reliable, good reasoning quality
- **O3**: Excellent reasoning but requires enhanced retry logic
- **Llama-3.1**: Local processing, requires GPU resources

Typical processing speed: 2-5 utterances/minute (depending on model and --cot setting)

## 🤝 Contributions
Contributions are welcome! Feel free to open an issue or submit a pull request. Areas of particular interest:
- Additional model backend support
- New annotation schemas
- Performance optimizations
- Integration with other CMC analysis tools

## 📜 Version History

- **v0.2.0** *(Current)*
  - **Corpus-agnostic**: One system prompt for all
  - **Improved politeness framework**: Added Brown & Levinson theoretical foundation

- **v0.1.0**  
  - **Multiple model support**: OpenAI (GPT-4o, O3) and local Llama-3.1
  - **Enhanced context**: Background summaries, thread starters, local conversational context
  - **Improved validation**: Better format checking and annotation reproducibility
  - **Quality assurance**: Comprehensive logging and checkpoint system

- **v0.0.1**  
  - Initial prototype with communicative act classification
  - Basic politeness/impoliteness tagging and meta-acts
  - CoT reasoning toggle and resumable run logic