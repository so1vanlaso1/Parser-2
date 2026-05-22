From `c:\Users\hi\Desktop\Praser`, run this in PowerShell.

**1. Create and activate a venv**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**2. Install dependencies**
Use the root [requirements.txt](/c:/Users/hi/Desktop/Praser/requirements.txt:1):

```powershell
pip install -r requirements.txt
```

If `torch` fails to install on your Python version, use Python 3.11 or 3.12 for the venv.

**3. Run the single-example pipeline**
```powershell
cd logic_pipeline
python scripts/run_one.py
```

The first run will download `Qwen/Qwen3.5-4B` from Hugging Face, so it may take a while and needs enough disk/RAM/GPU memory.

**4. Run batch parsing**
From inside `logic_pipeline`:

```powershell
python scripts/run_jsonl.py --limit 5
```

Full dataset:

```powershell
python scripts/run_jsonl.py
```

Start at a specific offset:

```powershell
python scripts/run_jsonl.py --start 10 --limit 20
```

Output goes to:

```text
logic_pipeline/artifacts/predictions.jsonl
```

**Optional: Ollama fallback**
The old backend still works if you want it:

```powershell
python scripts/run_jsonl.py --provider ollama --model qwen2.5:7b-instruct --limit 5
```

But for your requested Hugging Face model, use the default commands above.