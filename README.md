# Running the AO* proof search algorithm
The minimal necessary requirements.txt will be supplied later, but the conda & lean setup for copra suffices. Also put your OpenAI keys in `.secrets/openai_key.json`. Run
```bash
leanpkg configure
leanpkg build
```
which, among other things, ensures Lean 3 will be run even if you also have Lean 4 installed. Then you can simply run
```bash
python aostar_wrappers.py
```
