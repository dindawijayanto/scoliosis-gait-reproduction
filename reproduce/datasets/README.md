# Scoliosis1K data

| File | On GitHub | What |
|---|---|---|
| `Scoliosis1K_112.json`, `_114`, `_118`, `_1116` | yes | Train / test splits from OpenGait. The number is the positive : neutral : negative ratio in training. All configs use **`Scoliosis1K_118.json`** (1:1:8), the split of paper Table 4. |
| `Scoliosis1K-sil-pkl.zip` | no | The official silhouette release, password-protected. This is the file to upload to RunPod. |
| `Scoliosis1K-sil-pkl/` | no | The same data extracted: 1,493 children, `<id>/<class>/000_180/000_180.pkl`, each 300 frames of 64×64. |

The dataset is released under a signed agreement and must not be redistributed, so it is in `.gitignore`. To get it, sign the [release agreement](https://zhouzi180.github.io/Scoliosis1K/) and request the password from the authors, then:

```bash
unzip Scoliosis1K-sil-pkl.zip -d .      # asks for the password
```

Split 1:1:8 in numbers (children):

| | positive | neutral | negative | total |
|---|---|---|---|---|
| train | 74 | 74 | 596 | 744 |
| test | 419 | 126 | 204 | 749 |
