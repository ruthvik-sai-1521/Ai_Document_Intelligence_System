import pickle
from pathlib import Path
import sys

chunks_path = Path('embeddings/chunks.pkl')
if not chunks_path.exists():
    print("chunks.pkl does NOT exist")
    sys.exit(0)

with open(chunks_path, 'rb') as f:
    chunks = pickle.load(f)

print(f"Total indexed chunks: {len(chunks)}")

uid_counter = {}
for c in chunks:
    uid = c.get('metadata', {}).get('user_id')
    uid_str = str(uid)
    uid_counter[uid_str] = uid_counter.get(uid_str, 0) + 1

print(f"Unique user_ids in chunks: {list(uid_counter.keys())}")
for uid, count in uid_counter.items():
    print(f"  user_id={uid!r}: {count} chunks")

print("\nFirst 5 chunks:")
for i, c in enumerate(chunks[:5]):
    uid = c.get('metadata', {}).get('user_id')
    src = c.get('metadata', {}).get('source')
    print(f"  [{i}] user_id={uid!r} | source={src!r}")
