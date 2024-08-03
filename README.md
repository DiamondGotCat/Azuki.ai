
![Azuki](https://github.com/user-attachments/assets/2bcfd1d5-7998-4b3f-8e42-fe7269082d19)

Azuki.ai is a fully customizable AI.

Everything can be changed from the dataset state.

### What is this Naming!?
This name is, **from the Japanese** "あずき (Azuki)".

**"あずき" is "Red beans"** in the English.

## Why does Azuki.ai work?
1. Load GPT2 and that Tokenizer.
2. Add dataset for fine tuning, and Training Model using That.
3. Extract New Model.
4. Complete! This is all you need!

## Roadmap
- [x] Base Script and Structure
- [ ] Chat Plugin (Prompt Continue)
- [x] SM Model (Former name "XS")
- [ ] MD Model
- [ ] LG Model
- [ ] XL Model

And more!

## Require Spec
- SM Model: Can be run on some smartphones, and almost all PCs from 2015 onwards
- MD Model: Incomplete
- LG Model: Incomplete
- XL Model: Incomplete

## Latest default dataset for Azuki.ai
Please download from [This Repo](https://github.com/DiamondGotCat/Dataset-for-Azuki.ai)

## Dataset Contribute
**To make this project bigger, we need to make the dataset bigger.**
Please cooperate.

### NOTE
Divided the dataset into the following six categories:
- **Small** (sm) **:** A small, highly efficient dataset for mobile devices (e.g., generating sentence continuations)
- **Medium** (md) **:** A medium-sized, slightly smart dataset for low-spec PCs (e.g., solving common sense problems)
- **Large** (lg) **:** A large, smart dataset for medium-spec PCs (e.g., solving general problems)
- **Extra Large** (xl) **:** An extra-large, high-spec dataset for a Mac M1 or so (e.g., solving math problems for high school students)

## Files
- **execute.py:** Runner
- **training.py:** Training Script

## Customize Output
1. Download Latest Default Dataset
2. Edit data-{size}.json
3. Execute Training Script
