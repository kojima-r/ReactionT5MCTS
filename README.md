# ReactionT5-MCTS — 逆合成経路探索と PaRoutes ベンチマーク

**ReactionT5** の一段階逆合成予測を各ステップに用いた **モンテカルロ木探索 (MCTS)** による多段階逆合成経路プランナーです。
**PaRoutes** ベンチマーク（n1 / n5）上で性能を評価し、**AiZynthFinder** および文献値と比較します。

- 一段階モデル: [`sagawa/ReactionT5v2-retrosynthesis`](https://huggingface.co/sagawa/ReactionT5v2-retrosynthesis)
- ベンチマーク: [PaRoutes](https://github.com/MolecularAI/PaRoutes) (Genheden & Bjerrum, *Digit. Discov.* 2022)
- 探索: AiZynthFinder 型 MCTS（状態＝在庫外分子集合、行動＝1分子を1反応で展開）
- 乱数シード固定＋一段階予測の永続キャッシュにより **完全に再現可能**

---

## 1. ディレクトリ構成

```
ReactionT5MCTS/
├── README.md                  # 本ファイル
├── rt5mcts/                   # コアパッケージ
│   ├── model.py               #   ReactionT5 ラッパー（永続SQLiteキャッシュ・予算制御・シード固定）
│   ├── mcts.py                #   逆合成MCTS（PUCT選択・在庫貪欲ロールアウト・経路抽出）
│   ├── tree.py                #   経路木のデータ構造とヘルパー（mctsから分離）
│   ├── reward.py              #   差し替え可能な報酬（評価関数）: stock/yield/sa/composite/retrek
│   ├── yield_model.py         #   収率予測 ReactionT5v2-yield ラッパー（永続SQLiteキャッシュ）
│   ├── stock.py               #   在庫（購入可能分子）判定
│   ├── chem.py                #   RDKit ユーティリティ（正準化・断片分割）
│   └── reagents.py            #   一般的な試薬/溶媒/塩を「常時入手可能」として在庫に追加
├── run_reactiont5.py          # ReactionT5-MCTS 実行ドライバ（PaRoutes形式JSON出力）
├── run_aizynthfinder.py       # AiZynthFinder 実行ドライバ（aizynth環境で実行）
├── evaluate_routes.py         # PaRoutes 公式評価 route_quality.py 実行（aizynth環境で実行）
├── make_report.py             # md + html レポート生成
├── select_best.py             # sweep結果からベスト設定を選択
├── run_all.sh                 # 全工程オーケストレーション
├── setup_aizynth_env.sh       # AiZynthFinder 用 conda 環境の構築
├── configs/                   # MCTS ハイパーパラメータ設定（sweep用プリセット）
├── cache/                     # 一段階予測の永続キャッシュ（SQLite）
├── results/                   # 実行結果（routes/meta/eval JSON, STATUS.txt）
├── logs/                      # 各ステップのログ
├── reports/                   # report.md / report.html
├── aizynth_data/              # AiZynthFinder 公開モデル（download_public_data で取得）
├── paroutes/                  # PaRoutes リポジトリ（data/ にデータ配置済み）
└── test_reactiont5.py         # ReactionT5 の最小動作例（参照元）
```

---

## 2. 前提条件

- Linux + [conda / miniconda](https://docs.conda.io/en/latest/miniconda.html)
- PaRoutes データは `paroutes/data/` に配置済み（`n1/n5-targets.txt`, `n1/n5-stock.txt`, `n1/n5-routes.json` 等）
- GPU は不要（CPU で動作。ただし後述の通り CPU では低速）

本プロジェクトは **2 つの conda 環境**を使い分けます。

| 環境 | Python | 役割 | 主要パッケージ |
|---|---|---|---|
| `reactiont5` | 3.13 | ReactionT5-MCTS の実行 | torch, transformers, rdkit |
| `aizynth`    | 3.11 | AiZynthFinder の実行 **＋ PaRoutes 公式評価** | aizynthfinder, route-distances |

> 公式評価に必要な `route_distances` は Python < 3.9 を要求し `reactiont5` 環境に入らないため、
> `aizynth` 環境（依存として `route_distances` を含む）で評価を実行します。

---

## 3. 導入方法

### 3.1 `reactiont5` 環境（ReactionT5-MCTS 用）

```bash
conda create -n reactiont5 python=3.13 -y
conda activate reactiont5
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "transformers>=5" rdkit pandas numpy tqdm jinja2
```

初回実行時に ReactionT5 モデルが Hugging Face から自動ダウンロードされ、`~/.cache/huggingface/` にキャッシュされます。

### 3.2 `aizynth` 環境（AiZynthFinder ＋ 評価用）

```bash
bash setup_aizynth_env.sh
```

このスクリプトは以下を行います。
- `aizynth` conda 環境（Python 3.11）を作成
- `pip install "aizynthfinder[all]"`（`route-distances`, `rdkit`, `onnxruntime` 等を同時導入）

続いて AiZynthFinder の公開モデル（USPTO 展開ポリシー等）を取得します。

```bash
conda run -n aizynth download_public_data aizynth_data
```

> AiZynthFinder は上記の**公開 USPTO 展開ポリシ**を用い、在庫には **PaRoutes の在庫**
> (`paroutes/publication/aizynth_n{1,5}_stock.txt`) を使います（文献の PaRoutes 専用モデルとは異なる点に注意）。

---

## 4. 使い方

### 4.1 一括実行（推奨）

全工程（ハイパラsweep → 公式評価 → ベスト設定選択 → n1/n5 本実行 → AiZynthFinder → 評価 → レポート）を実行します。

```bash
bash run_all.sh
```

- 進捗は `results/STATUS.txt` に、各ステップの詳細は `logs/` に記録されます。
- 一段階予測はキャッシュ（`cache/rt5_b5.sqlite`）に保存され、**中断しても再実行で自動再開**します。
- 完了すると `reports/report.md` と `reports/report.html` が生成されます。

`run_all.sh` 冒頭の変数で規模を調整できます。

```bash
SWEEP_N=3        # sweep に使う n1 ターゲット数
MAIN_N=8         # 本実行のターゲット数（n1/n5 各）
SWEEP_BUDGET=15  # sweep の1ターゲットあたり新規モデル呼び出し上限
MAIN_BUDGET=20   # 本実行の同上
MAXBEAMS=5       # ビーム幅
SEED=42          # 乱数シード
SWEEP_CONFIGS="base ew3 it200 c0p5 c3p0 rprob md8"   # sweep する設定
```

### 4.2 個別実行

```bash
# (1) ReactionT5-MCTS を n1 の8ターゲットで実行
conda run -n reactiont5 python run_reactiont5.py \
    --route-set n1 --n-targets 8 --config configs/base.json \
    --tag best --out-dir results/rt5 --cache cache/rt5_b5.sqlite \
    --max-beams 5 --model-budget 20 --seed 42

# (2) 公式評価（aizynth 環境）
conda run -n aizynth python evaluate_routes.py \
    --routes results/rt5/routes_best_n1.json \
    --references paroutes/data/n1-routes.json --n 8 \
    --method ReactionT5-MCTS --tag best --route-set n1 \
    --out results/eval/rt5_best_n1.json

# (3) AiZynthFinder を同じ条件で実行（aizynth 環境）
conda run -n aizynth python run_aizynthfinder.py \
    --route-set n1 --n-targets 8 --out-dir results/aizynth

# (4) レポート生成
conda run -n reactiont5 python make_report.py
```

---

## 5. MCTS ハイパーパラメータ

`configs/*.json` で指定します（`run_reactiont5.py --config` で読み込み）。

| キー | 説明 | base |
|---|---|---|
| `expansion_width` | 1回の展開で生成する候補反応数 | 5 |
| `iterations` | MCTS の反復回数 | 100 |
| `c_puct` | PUCT の探索係数 | 1.4 |
| `max_depth` | 経路の最大反応段数 | 12 |
| `rollout_width` | ロールアウトで見る候補数（prob方策時） | 1 |
| `rollout_policy` | `stock`＝在庫内反応物を最大化 / `prob`＝確率top-1 | stock |
| `model_budget` | 1ターゲットあたり新規モデル呼び出し上限（`--model-budget`で上書き可） | 20 |

---

## 6. 報酬（評価関数）の設定

MCTS が「良い経路」を判断する **評価関数（報酬）は差し替え可能** です。報酬は経路木を
`[0, 1]` のスカラー（大きいほど良い）に写す関数で、(1) ロールアウトのバックアップ値、
(2) 解けた経路に保存されるスコア、(3) 最終ランキング、の 3 か所に使われます。報酬を変える
だけで、探索本体に手を入れずにプランナの最適化目標を変更できます（実装: `rt5mcts/reward.py`）。

報酬は **単体クラス** か、**複数の「項(term)」を重み付きで組み合わせた合成報酬** のいずれかです。

### 6.1 組み込みの報酬（`reward_policy`）

| policy | 内容 |
|---|---|
| `stock` | 従来の値（`reward_depth_decay^反応数 × 在庫葉の割合`）。既定。 |
| `yield` | `stock` × （経路の集約収率）^`yield_weight`。ReactionT5v2-yield で各ステップの収率を予測。 |
| `sa` | `合成容易性(SA) × 在庫割合 × 浅さ` の積プリセット。 |
| `composite` | 任意の項を重み付きで結合する汎用報酬（`reward_terms` / `reward_combine`）。 |
| `retrek` | ReTReK 型の 6 知識スコアを重み付き**和**で結合するプリセット（下記）。 |

### 6.2 報酬項（term）と ReTReK 6 スコア

`composite` / `retrek` は以下の項を `reward_terms={項:重み}` で自由に組み合わせられます
（各項は経路→`[0,1]`、反応ステップ平均。反応が無い経路は中立値 1.0）。

| 項名 | 意味 |
|---|---|
| `in_stock` | 葉分子のうち在庫（購入可能）な割合 |
| `solved` | 全葉が在庫なら 1 |
| `shallow` | `reward_depth_decay^反応数`（短い経路を優先） |
| `yield` | 各ステップの予測収率の集約（`yield_agg`: geomean/product/min/mean） |
| `sa` | 葉分子の合成容易性（RDKit SA、易しいほど高） |
| **`cdscore`** | **Convergent Disconnection**: 反応物断片が同サイズ＝収束的なほど高（`min/max` 重原子数） |
| **`asscore`** | **Available Substances**: 在庫（購入可能）葉の割合 |
| **`rdscore`** | **Ring Disconnection**: 環を形成する反応（生成物の環数 > 反応物合計）なら高 |
| **`stscore`** | **Selective Transformation**: 反応物が少ない（選択的）ほど高（`1/反応物数`） |
| **`intermediate`** | **Intermediate**: 経路中間体の妥当さ（SA ベース） |
| **`template`** | **Template**: 各反応に一段階モデルが与えた確率（well-precedented ほど高） |

> ReTReK 6 スコア（cdscore, asscore, rdscore, stscore, intermediate, template）は公開論文の
> 定義に基づく **独自実装** です（外部コードは参照していません）。加算的定式化に対応するため
> `retrek` プリセットは `reward_combine="sum"`（重み付き平均）を既定とします。

`reward_combine` は `product`（重み付き幾何積、`composite`/`sa` の既定）または
`sum`（重み付き平均、`retrek` の既定）。空にすると各報酬の既定を使います。

### 6.3 設定方法（4 通り）

1. **CLI**（`run_reactiont5.py`）:
   ```bash
   # 収率報酬
   --reward-policy yield --yield-weight 1.0 --yield-agg geomean

   # ReTReK 6 スコア（プリセット）
   --reward-policy retrek

   # 任意の項を重み付きで（部分集合も可）
   --reward-policy composite --reward-combine sum \
     --reward-terms '{"in_stock":1,"shallow":1,"yield":1,"cdscore":1,"rdscore":0.5}'
   ```
2. **config JSON**（`--config`）: `reward_policy` / `reward_terms` / `reward_combine` /
   `yield_weight` / `yield_agg` / `reward_depth_decay` を記述。例: `configs/base_yield.json`,
   `configs/base_composite.json`, `configs/base_retrek.json`。
3. **`make_reward()`**: `from rt5mcts.reward import make_reward; make_reward("retrek")`。
4. **直接注入**: `RetroMCTS(model, stock, cfg, reward=<任意のcallable>)`。

> 収率を使う報酬（`yield`、または `yield` 項を含む合成報酬）では収率モデル
> `sagawa/ReactionT5v2-yield` を自動構築し、予測は `--yield-cache`（既定
> `cache/rt5_yield.sqlite`）に永続キャッシュされます。不要な場合は構築されません。

### 6.4 新しい報酬・項の追加

3 行で追加できます（自動でレジストリに登録され、`reward_policy` / `reward_terms` から利用可能）。

```python
from rt5mcts.reward import RouteReward, register, term, RewardContext

@term("my_term")                      # 新しい項: 経路 -> [0,1]
def my_term(root, ctx: RewardContext): return ...

@register                             # 新しい報酬クラス
class MyReward(RouteReward):
    name = "mine"
    def score(self, root): return ...
```

---

## 7. 出力

| パス | 内容 |
|---|---|
| `results/rt5/routes_<tag>_<set>.json` | ReactionT5-MCTS の予測経路（PaRoutes形式） |
| `results/rt5/meta_<tag>_<set>.json`   | solve率・in-stock frac・実行時間・設定 |
| `results/aizynth/routes_azf_<set>.json` | AiZynthFinder の予測経路 |
| `results/eval/*.json` | PaRoutes 公式指標（solved / top-1,5,10） |
| `reports/report.md`, `reports/report.html` | 比較表付きレポート |

---

## 8. 手法のポイント

- **一段階予測キャッシュ**: 正準SMILES＋ビーム幅をキーに SQLite へ保存。同一分子はモデルを1回だけ評価するため、
  sweep 全体・再実行で高速かつ結果が一貫。
- **試薬在庫の追加** (`rt5mcts/reagents.py`): ReactionT5 は USPTO 反応で学習しており試薬/溶媒/塩を反応物として出力します。
  PaRoutes の在庫は building block のみなので、そのままでは経路が閉じません。一般的な試薬 99 種を「常時入手可能」として
  在庫に追加します（化学的に妥当）。`--no-augment-stock` で無効化可能。
- **在庫貪欲ロールアウト**: ロールアウト時、キャッシュ済みの候補（追加のモデル呼び出し不要）から
  「在庫内反応物を最大化する切断」を選び、経路が在庫に到達しやすくします。
- **in-stock frac**: 「全葉が在庫」という厳密 solve とは別に、経路の葉のうち在庫に到達した割合の最大値を記録。
  1.0 に近いほど「あと数葉で完成」を意味し、厳密 solve が隠す進捗を可視化します。

---

## 9. 結果（初回実行: 各8ターゲット, beam=5, budget=20, max_depth=12, ベスト設定=base）

| 手法 | セット | solved | in-stock frac | top-1/5/10 | 時間/ターゲット |
|---|---|---|---|---|---|
| 文献 MCTS (PaRoutes v2.0) | n1 | 9716/10000 | – | 0.237 / 0.511 / 0.541 | N/A |
| 文献 MCTS (PaRoutes v2.0) | n5 | 9689/10000 | – | 0.124 / 0.358 / 0.406 | N/A |
| **AiZynthFinder**（本実験） | n1 | 2/8 | – | 0 / 0 / 0 | 約39秒 |
| **AiZynthFinder**（本実験） | n5 | 3/8 | – | 0 / 0 / 0 | 約39秒 |
| **ReactionT5-MCTS**（本実験） | n1 | 1/8 | 0.824 | 0 / 0 / 0 | 約4.4分 |
| **ReactionT5-MCTS**（本実験） | n5 | 2/8 | 0.947 | 0 / 0 / 0 | 約9.5分 |

ハイパラ sweep の主な知見: `max_depth=8` は 0/3（浅すぎて解けない）、確率ロールアウトは在庫貪欲より劣る（0/3 vs 1/3）。

---

## 10. 制約・注意点

- **CPU では低速**: n1 の創薬分子1件のビーム探索に約30〜45秒。PaRoutes のターゲットは8〜10段の深い経路のため、
  全10000ターゲットの評価は非現実的です。本リポジトリでは各セット少数（既定8）ターゲットに絞り、
  1ターゲットあたりの新規モデル呼び出し回数を予算で制限した **計算制約下の公正比較** としています。
- **top-k はほぼ0**: ReactionT5 はテンプレートフリーの seq2seq モデルで、参照経路（テンプレートベースで生成）と
  木構造まで完全一致することは稀です。従って本評価では **solve率と in-stock frac を主指標**とし、top-k は参考値です。
- **AiZynthFinder** は公開 USPTO ポリシ＋PaRoutes 在庫を使用（文献の PaRoutes 専用モデルではない）。
- 評価時、`route_distances` の木編集距離は反応物7個までしか扱えないため、`evaluate_routes.py` で
  8個以上を恒等順序にフォールバックするパッチを適用しています（solve判定には影響しません）。

---

## 11. 再現性

- すべての実行で乱数シードを固定（`--seed 42`）。ReactionT5 のビーム探索は決定的で、一段階予測はキャッシュされます。
- GPU 非使用のため数値演算の非決定性もありません。
- 同じキャッシュ・同じ設定なら `bash run_all.sh` は同一の結果を再現します。

## 参考文献

Genheden, S.; Bjerrum, E. *PaRoutes: Towards a Framework for Benchmarking Retrosynthesis Route Predictions.*
Digit. Discov. 2022, 1 (4), 527–539. https://doi.org/10.1039/D2DD00015F
