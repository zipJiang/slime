"""Keep native history intact while excluding rejected read requests from SFT loss."""
from synthetic_data import export as export_all

def rejected_read(turn):
 return turn.tag=='task' and turn.transition is not None and any(
  str(m.get('content','')).startswith('error: lines must be between 1 and 60')
  for m in turn.transition.messages)

def export(turns,q,source,first_fold):
 ignored=[i for i,t in enumerate(turns) if i>=first_fold and rejected_read(t)]
 selected=tuple(t for i,t in enumerate(turns) if i>=first_fold and i not in ignored)
 rows=export_all(selected,q,source,0)
 for r in rows:r['metadata']['ignored_rejected_read_turns']=ignored
 return rows
