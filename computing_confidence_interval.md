## Uncertainty Analysis for the GPD

裾を推定するほとんどのEVT法は、特定の閾値 $t$ 以降でのEVIの安定性を前提としている[25, 15]。しかし、裾に低密度混合物が存在する場合、情報量が少ないために閾値が高いとEVIが不安定になり、誤算される可能性がある。第3節では、低密度混合物の存在により選択された閾値が低くなると、この誤算がpWCET推定値に悪影響を与える可能性があることを示した。裾の低密度混合物を検出し、このようなシナリオを回避するために、本節ではEVIの推定の原理と、裾の低密度混合物に近づくにつれて推定値がどのように変化するかについて説明する。実際、この変化をロバストな方法で検出し定量化することが本論文の目標であり、TailIDの中核原理である。

### 4.1 Maximum Likelihood Estimation and Uncertainty Analysis

確率変数 $X$ の観測値と、式1で $Y_{t}:=(X-t \mid X>t)$ として定義されている閾値 $t$ を超える確率変数 $X$ を考慮すると、GPDのパラメータ $\xi$ を推定できる。式に関連するモデルの不確実性に加えて、 $\xi$ の推定には、入力 $y_{i}$ の変動性によるサンプリングの不確実性も含まれる。GPDの推定EVI $\hat{\xi}$ の不確実性を評価するには、信頼区間を計算する必要がある。信頼区間は、真のパラメータ $\xi$ が指定された信頼水準 $\gamma$ の範囲内に含まれる可能性のある値の範囲を示す。

推定値の信頼区間を計算する古典的な方法は、誤差の漸近分布を導出することである。すなわち、パラメータの真の値と推定値の差は、ある分布に従うということである。推定値が最尤推定法 (MLE)から導かれる場合、その差は次のようにガウス分布に収束する。

定理9 (推定量の漸近正規性[49])

$\xi$ をパラメータの真の値、 $\hat{\xi}_{\text {mle }}$ を $\xi$ の最大尤度推定値、 $n$ を標本数とすると、以下の式が成り立つ。
\[
\sqrt{n}\left(\hat{\xi}_{m l e}-\xi\right) \xrightarrow{d} \mathcal{N}(0, \operatorname{var}(\xi)), \quad (4)
\]
真の値とパラメータの推定値の差は、ガウス分布 $\mathcal{N}(0, \operatorname{var}(\xi))$ に収束する。

この定理は、主要な確率論的結果、具体的には大数の法則と中心極限定理を組み合わせたものである。式4は、多くの基礎的な確率論および統計学の書籍[49][32]に記載されている古典的な結果である。式4を展開するには、極値指数 $\hat{\xi}_{\text {mle }}$ とその分散 $\operatorname{var}(\xi)$ の最尤推定値を求める必要がある。そして、式4を用いて $\xi$ の推定値の信頼区間を計算できる。

まず、式4で用いられている手法である最大尤度推定法 (MLE)[35]をパラメータ推定に導入する。分布のパラメータ、特にGPDのEVI (尤度関数) $\xi$ を推定する最も一般的な方法の一つがMLE [35]である。MLEは、観測データの集合が与えられた場合に尤度関数を最大化するパラメータ値を見つけるために用いられる統計的手法である。パラメータ $\theta$ を持つ分布の標本 $X=\left\{x_{i}\right\}_{1 \leq i \leq n}$ の尤度関数 $L(\theta \mid X)$ は、次のように表される。

定義10 (尤度)。確率密度関数 $f$ を持つ、サイズ $n$ の確率変数の標本を $x_{1}, \ldots, x_{n}$ とすると、尤度関数は次のように定義される。

\[
L(\theta \mid X)=\prod_{i=1}^{n} f\left(\theta \mid x_{i}\right)
\]
ここで、 $f\left(\theta \mid x_{i}\right)$ は、パラメータ $\theta$ が与えられたときの $x_{i}$ における分布の確率密度関数 (PDF)である。

ここで、対数尤度は次のように定義される。

定義11 (対数尤度)。

パラメータ $\theta$ を与えられたときの、 $x_{i}$ におけるPDFを $f\left(\theta \mid x_{i}\right)$ とすると、
\[
\ell(\theta \mid X)=\sum_{i=1}^{n} \log f\left(\theta \mid x_{i}\right)
\]
パラメータ $\theta$ を与えられた対数尤度関数である。
 $\theta$ の MLE ( $\hat{\theta}$ と表記)は、次の式を満たす値である。

定義12 (最大尤度推定)

対数尤度関数 $\ell(\theta \mid X)$ が与えられたとき、 $\hat{\theta}_{\text {mle }}$ は対数尤度を最大化するパラメータ $\theta$ の値である。すなわち、次の式で表される。
\[
\hat{\theta}_{m l e}=\arg \max _{\theta} \ell(\theta \mid X)
\]

これで、GPD の EVI の MLE を証明するツールが手に入りました。

### 4.2 Computing Confidence Intervals on the Extreme Value Index

ここで、上記の定義をGPDの特定のケースに適用する。GPDの場合、 $x_{1}, \ldots, x_{n}$ の観測値と、与えられた閾値 $t$ 、 $Y_{t}=\left\{y_{i}\right\}_{1 \leq i \leq n}$ に対する超過分 $Y_{t}$ を考慮すると、GPDの尤度関数は次のようになる。
\[
L_{g p d}\left(\xi, \sigma \mid Y_{t}\right)=\prod_{i=1}^{n} f_{g p d}\left(\xi, \sigma \mid y_{i}\right)=\prod_{i=1}^{n} \frac{1}{\sigma}\left(1+\frac{\xi y_{i}}{\sigma}\right)^{-\frac{1}{\xi}-1}
\]
対数尤度関数は次の形になる。
\[
\ell_{g p d}\left(\xi, \sigma \mid Y_{t}\right)=-n \log \sigma-\left(1+\frac{1}{\xi}\right) \sum_{i=1}^{n} \log \left(1+\frac{\xi y_{i}}{\sigma}\right)
\]

変数 $\psi=\frac{\sigma}{\xi}$ の変更点を見てみよう。
\[
\ell_{g p d}\left(\xi, \sigma \mid Y_{t}\right)=-n \log (\xi \psi)-\left(1+\frac{1}{\xi}\right) \sum_{i=1}^{n} \log \left(1+\frac{y_{i}}{\psi}\right)
\]

MLE を実行して $\hat{\xi}$ を取得する:
\[
\frac{\partial \ell_{g p d}\left(\xi, \sigma \mid Y_{t}\right)}{\partial \xi}=-\frac{n \psi}{\xi}+\frac{1}{\xi^{2}} \sum_{i=1}^{n} \log \left(1+\frac{y_{i}}{\psi}\right)=0 .
\]

これは次のことにつながる:
\[
\hat{\xi}_{m l e}=\frac{1}{n} \sum_{i=1}^{n} \log \left(1+\frac{y_{i}}{\psi}\right)
\]

極値指数 $\xi$ は完全に分離できないため、この式は数値的に解く必要がある。これを実現するために、ismevパッケージ[21]のgpd.fit()関数を用いて、MLEを用いてGPDモデルを閾値超過に適合させる。

パラメータ $\theta, \operatorname{var}(\theta)$ の分散を計算するために、Cramér-Rao境界を用いることができる。これは、確率変数 $X$ の観測値に基づいて、分散が最小となる推定値を求めることで不確実性を低減するものである[39]。Cramér-Rao境界は以下のように表される。

定理13 (クラメールラオの境界)

$\theta$ を不偏推定量 $\hat{\theta}$ を持つパラメータとすると、推定量の分散は次の式で制限される。
\[
\operatorname{var}(\hat{\theta}) \geq \frac{1}{I(\theta)} \quad (13)
\]
ここで、 $I(\theta)$ はフィッシャー情報量である。
フィッシャー情報量は以下のように定義される。

定義14 (フィッシャー情報量)

パラメータ $\theta$ を与えられた尤度関数 $\ell(\theta \mid X)$ とすると、
\[
I(\theta)=-E\left[\frac{\partial^{2}}{\partial \xi^{2}} \ell(\theta \mid X)\right]
\]
これは、MLE推定値の導関数の期待値を用いる。この数学的枠組みを用いることで、式4で導入した極値指標 $\xi$ のMLE周エッジの不確実性を計算できる。

式13に示した手順に従うと、EVIのCramér-Rao境界は
\[
\operatorname{var}(\hat{\xi}) \geq \frac{1}{I_{1}(\xi)}
\]

次に、 $\sigma$ が既知の場合のフィッシャー情報は次のように計算される。
\[
I_{1}(\hat{\xi})=-E\left(\frac{\partial^{2} \ell_{g p d}\left(\xi, \sigma \mid Y_{t}\right)}{\partial \xi^{2}}\right)=-E\left(\frac{1}{\hat{\xi}^{2}}-\frac{2}{\hat{\xi}^{3}} \hat{\xi}\right)=\frac{1}{\hat{\xi}^{2}}
\]

この結果を式13に代入すると、
\[
\operatorname{var}(\hat{\xi}) \geq \frac{1}{I_{1}(\hat{\xi})}=\hat{\xi}^{2}
\]

最後に、GPDの場合の式4を完成させる。
\[
\sqrt{n}(\hat{\xi}-\xi) \xrightarrow{d} \mathcal{N}\left(0, \xi^{2}\right)
\]

式18により、EVIの推定のための信頼区間を計算できる。

- 例15. 式12を用いて推定パラメータ $\hat{\xi}_{\text {mle }}=1$ を持つ観測値 $y_{1}, \ldots, y_{100}$ について考える。信頼水準 $\gamma=0.95$ における $\xi$ の信頼区間を計算する。式18によれば、以下の式が得られる。
\[
\sqrt{100}\left(\hat{\xi}_{m l e}-\xi\right) \xrightarrow{d} \mathcal{N}\left(0,1^{2}\right)
\]

信頼区間を構築するために、 $\alpha=1-\gamma$ を定義し、 $Z \sim \mathcal{N}(0,1.$ が以下を満たす。
\[
P\left(-z_{\alpha / 2} \leq Z \leq z_{\alpha / 2}\right)=1-\alpha=\gamma
\]
ここで、 $z_{\frac{\alpha}{2}}$ は標準正規分布の($1-\alpha / 2$)-分位数を表す。 $\alpha=0.05$ 、 $z_{\alpha / 2} \approx 1.96$ の場合は、 $z_{\frac{\alpha}{2}}$ が分位数となる。

式19の標準化を用いると、次の式が得られる。
\[
P\left(-z_{\alpha / 2} \leq 10\left(\hat{\xi}_{m l e}-\xi\right) \leq z_{\alpha / 2}\right)=\gamma
\]

不等式を整理すると、次のようになる。
\[
P\left(\hat{\xi}_{m l e}-\frac{z_{\alpha / 2}}{10} \leq \xi \leq \hat{\xi}_{m l e}+\frac{z_{\alpha / 2}}{10}\right)=\gamma .
\]

したがって、 $\xi$ の0.95信頼区間は次のように与えられる。
\[
\xi \in\left[\hat{\xi}_{m l e}-\frac{z_{\alpha / 2}}{10}, \hat{\xi}_{m l e}+\frac{z_{\alpha / 2}}{10}\right]=[0.804,1.196]
\]
 $\hat{\xi}_{\text {mle }}=1$ と $z_{\alpha / 2} \approx 1.96$ を思い出してください。
MLE を使用しているため、推定値の変動性を捉えながら、モデルの不確実性を可能な限り低減できる。

最後に、式18の表現はTailIDアルゴリズムの中核であり、尾部に変化があるかどうかを評価するために使用される。
