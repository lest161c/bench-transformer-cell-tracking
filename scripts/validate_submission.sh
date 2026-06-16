#!/bin/bash
# Run locally before sbatch to catch import/syntax/arg errors early.
# Usage: cd bench-transformer-cell-tracking && bash scripts/validate_submission.sh

set -e
VENV="../trackastra/.venv"
ACTIVATE="$VENV/bin/activate"
PY="python3"

if [ -f "$ACTIVATE" ]; then
    . "$ACTIVATE"
elif command -v uv &>/dev/null; then
    PY="uv run python3"
fi

echo "=== Pre-submission validation ==="

# 1. Python syntax check all touched files
echo -n "1. Syntax check... "
for f in \
    trackastra/trackastra/model/model.py \
    trackastra/trackastra/model/dino_encoder.py \
    trackastra/trackastra/model/ssl_dino_trainer.py \
    configs/ssl_dino_pretrain.yaml; do
    if [[ $f == *.py ]]; then
        $PY -c "import ast; ast.parse(open('$f').read())" 2>/dev/null || {
            echo "FAILED: $f has syntax errors"; exit 1
        }
    fi
done
echo "OK"

# 2. Can the model be instantiated with use_dino=True?
echo -n "2. Model init (use_dino=True)... "
$PY -c "
import sys; sys.path.insert(0, 'trackastra')
from trackastra.model import TrackingTransformer
m = TrackingTransformer(coord_dim=2, feat_dim=0, d_model=64, nhead=2,
                        num_encoder_layers=2, num_decoder_layers=2, use_dino=True)
assert hasattr(m, 'dino_proj'), 'missing dino_proj'
assert hasattr(m, 'dino_pos_proj'), 'missing dino_pos_proj'
assert m.config['use_dino'] == True
" 2>&1 || { echo "FAILED"; exit 1; }
echo "OK"

# 3. Model init WITHOUT dino (backward compat)
echo -n "3. Model init (use_dino=False)... "
$PY -c "
import sys; sys.path.insert(0, 'trackastra')
from trackastra.model import TrackingTransformer
m = TrackingTransformer(coord_dim=2, feat_dim=7, d_model=64, nhead=2,
                        num_encoder_layers=2, num_decoder_layers=2, use_dino=False)
assert not hasattr(m, 'dino_proj'), 'should not have dino_proj'
" 2>&1 || { echo "FAILED"; exit 1; }
echo "OK"

# 4. Can ssl_dino_trainer import without errors?
echo -n "4. SSL trainer import... "
$PY -c "
import sys; sys.path.insert(0, 'trackastra')
from trackastra.model.ssl_dino_trainer import SSLDinoDataset, nt_xent_loss, train_ssl
" 2>&1 || { echo "FAILED"; exit 1; }
echo "OK"

# 5. Config YAML is valid with correct types
echo -n "5. Config types... "
$PY -c "
import yaml
with open('configs/ssl_dino_pretrain.yaml') as f:
    cfg = yaml.safe_load(f)
assert cfg['use_dino'] == True
s = cfg['ssl']
assert isinstance(s['epochs'], int), f'epochs type={type(s[\"epochs\"])}'
assert isinstance(s['lr'], float), f'lr type={type(s[\"lr\"])} value={s[\"lr\"]!r}'
assert isinstance(s['temperature'], float), f'temp type={type(s[\"temperature\"])}'
assert isinstance(s['batch_size'], int), f'batch_size type={type(s[\"batch_size\"])}'
" 2>&1 || { echo "FAILED"; exit 1; }
echo "OK"

# 6. End-to-end: load config + instantiate model (full path match)
echo -n "6. End-to-end model + config... "
$PY -c "
import yaml, sys; sys.path.insert(0, 'trackastra')
from trackastra.model import TrackingTransformer
with open('configs/ssl_dino_pretrain.yaml') as f:
    cfg = yaml.safe_load(f)
m = TrackingTransformer(
    coord_dim=cfg.get('ndim', 2), feat_dim=0,
    d_model=cfg.get('d_model', 320), nhead=cfg.get('nhead', 4),
    num_encoder_layers=cfg.get('num_encoder_layers', 6),
    num_decoder_layers=cfg.get('num_decoder_layers', 6),
    dropout=cfg.get('dropout', 0.01), window=cfg.get('window', 10),
    pos_embed_per_dim=cfg.get('pos_embed_per_dim', 32),
    feat_embed_per_dim=cfg.get('feat_embed_per_dim', 8),
    use_dino=cfg.get('use_dino', False),
)
assert hasattr(m, 'dino_proj')
# Verify AdamW would accept the lr
s = cfg['ssl']
lr = float(s['lr'])
assert isinstance(lr, float) and lr > 0
print(f'  lr={lr} temp={s[\"temperature\"]} epochs={s[\"epochs\"]} bs={s[\"batch_size\"]}')
" 2>&1 || { echo "FAILED"; exit 1; }
echo "OK"

# 7. Slurm script has no obvious issues
echo -n "7. Slurm syntax... "
if head -1 benchmark_combined/run_ssl_dino_pretrain.slurm | grep -q "#!/bin/bash"; then
    echo "OK"
else
    echo "FAILED: missing shebang"
    exit 1
fi

echo ""
echo "=== All validations passed. Safe to submit. ==="
