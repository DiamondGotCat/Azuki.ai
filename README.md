
![Azuki](https://github.com/user-attachments/assets/2bcfd1d5-7998-4b3f-8e42-fe7269082d19)

Azuki.ai is a fully customizable AI.

Everything can be changed from the dataset state.

### We need your help
This is a solo build, so I'll probably run out of steam soon.

Someone please help: extend the dataset, suggest new features, implement new features.

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
- [ ] Essential Plugin (LLM can using External Compute. e.g., Calc using Python, Assist your Coding and more!)
- [ ] Chat Model (for Chat Plugin)
- [ ] CLI Optimize
- [ ] GUI Mode
- [ ] Chat Server Mode (e.g., Assist your Coding with VSCode, Using Azuki.ai Server)
- [ ] Your Document Import
- [ ] Image Plugin (Seeing Image)
- [ ] Chat Plugin (Prompt Continue)
- [x] SM Model (Former name "XS")
- [ ] JP Model
- [x] MD Model
- [ ] LG Model
- [ ] XL Model
- [x] CD Model

And more!

## Require Spec
- SM Model: Can be run on some smartphones, and almost all PCs from 2015 onwards
- CD Model: Unknown
- MD Model: Unknown
- LG Model: Incomplete
- XL Model: Incomplete
- JP Model: Incomplete

## We need stars!
If you like it, please click the star right away.

It will help us spread our project to more people.

[![Star History Chart](https://api.star-history.com/svg?repos=DiamondGotCat/Azuki.ai&type=Date)](https://star-history.com/#DiamondGotCat/Azuki.ai&Date)

## Latest default dataset for Azuki.ai
Please download from [This Repo](https://github.com/DiamondGotCat/Dataset-for-Azuki.ai)

## Dataset Contribute
**To make this project bigger, we need to make the dataset bigger.**
Please cooperate.

### NOTE
Divided the dataset into the following five categories:
- **Small** (sm) **:** A small, highly efficient dataset for mobile devices (e.g., generating sentence continuations)
- **Code** (cd) **:** Python Knowledge (Small Model for Coding Assistant)
- **Medium** (md) **:** A medium-sized, slightly smart dataset for low-spec PCs (e.g., solving common sense problems)
- **Large** (lg) **:** A large, smart dataset for medium-spec PCs (e.g., solving general problems)
- **Extra Large** (xl) **:** An extra-large, high-spec dataset for a Mac M1 or so (e.g., solving math problems for high school students)
- **Japanese** (jp) **:** Japanese Model

## Files
- **execute.py:** Runner
- **training.py:** Training Script

## Customize Output
1. Download Latest Default Dataset
2. Edit data-{size}.json
3. Execute Training Script
