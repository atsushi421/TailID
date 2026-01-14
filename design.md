# Design

提案されたアプローチであるTailIDは、テールにおける低密度混合がテールの挙動の変化を引き起こし、それがEVI $\xi$ の変化として現れるという事実に基づいている。TailIDは、第4節で示した理論的展開に基づいている。特に、アルゴリズムの中核は、式18におけるGPDのEVIのMLEの信頼区間である。

TailIDアルゴリズムは、EVI $\xi$ の変化を引き起こす可能性のある極値を、テール挙動の変化が検出されるか、全てのポイントが処理されるまで、1つずつ反復的に分析する。 $\xi$ の推定値が信頼区間外になった場合、推定値が不適切であることを意味するのではなく、検出された混合成分によって挙動が変化したことを意味するため、テール推定において混合成分の存在を考慮した、より微妙なテール分析を適用する必要があることに注意。

アルゴリズム 1 で詳述されている TailID は、サンプルデータ $X$ 、信頼区間 $\gamma$ 、極値パーセンタイル $p_{M}$ 、候補ポイントのパーセンタイル $p_{c_{1}}$ に基づいて動作し、サンプルデータが ID であるという仮説に対してより敏感なポイントのセット $S$ を返す。敏感なポイントとは、サンプル内で考慮された場合、それらなしで計算された信頼区間から EVI が外れてしまう可能性があるポイントである。また、TailID は、十分に高いパーセンタイルのポイント $p_{c}$ を選択することで、最後のコンポーネント (ID 仮定を破るポイントの前) を裾の推定に考慮するため、混合内の複合分布の数には依存しない。セクション 5.1 では、最後のコンポーネントを検出するために $p_{c}$ を設定する方法を示す。

```
%Algorithm 1
\begin{algorithm}[t]
\caption{Detection of Tail ID-Sensitive Points}
\KwIn{$X$: Sample data for analysis\\
$p_M$: Extreme value percentile\\
$p_{c_1}$: Candidate percentile\\
$\gamma$: Confidence level}
\KwOut{$S$: Data points inconsistent with the tail’s identical distribution assumption}

\SetKwFunction{TailID}{TailID}
\SetKwFunction{Quantile}{Quantile}
\SetKwFunction{SelectCandidates}{SelectCandidates}
\SetKwFunction{ExcessSet}{ExcessSet}
\SetKwFunction{fitGPD}{fitGPDEVI}
\SetKwFunction{ComputeCI}{ComputeGPDCI}
\SetKwFunction{IsEmpty}{IsEmpty}
\SetKwFunction{Add}{Add}

%Algorithm Starts
\Fn{\TailID{$X,p_M,p_{c_1},\gamma$}}{
Initialize empty list $S$\;
$t_M \leftarrow$ \Quantile{$X,p_M$}\;
$C \leftarrow$ \SelectCandidates{$X,p_{c_1}$}\;
$X^0 \leftarrow X \setminus C$\;
$Y_{t_M}^0 \leftarrow$ \ExcessSet{$X^0,t_M$}\;
$\xi_{t_M}^0 \leftarrow$ \fitGPD{$Y_{t_M}^0$}\;
$I_{t_M}^0 \leftarrow$ \ComputeCI{$\xi_{t_M}^0,\gamma$}\;

\For{$i \in \{1,\ldots,\text{length}(C)\}$}{
    \If{\IsEmpty{$S$}}{
        $X^i \leftarrow X^{i-1} \cup c_i$\;
        $Y_{t_M}^i \leftarrow$ \ExcessSet{$X^i,t_M$}\;
        $\xi_{t_M}^i \leftarrow$ \fitGPD{$Y_{t_M}^i$}\;
        \If{$\xi_{t_M}^i \notin I_{t_M}^{i-1}$}{
            $S \leftarrow$ \Add{$S,c_i$}\;
        }
        \Else{
            $I_{t_M}^i \leftarrow$ \ComputeCI{$\xi_{t_M}^i,\gamma$}\;
        }
    }
    \Else{
        $S \leftarrow$ \Add{$S,c_i$}\;
    }
}
\Return{$S$}\;
}
\end{algorithm}

```

このアルゴリズムはまず、感度点が格納される $S$ データ構造 (2行目)を初期化する。時系列 $X=\left(x_{i}\right)_{1 \leq i \leq n}$ が与えられた場合、実装されたアルゴリズムはPoT法を適用して分布の裾を選択する。PoTの閾値 $t_{M}$ は、極値パーセンタイル $p_{M}$ に基づいて計算され、 $X$ のデータの $p_{M} \cdot 100$ パーセントが $t_{M}$ を下回る、すなわち $t_{M}:=$ の分位点 $\left(X, p_{M}\right)$ となるようにする (3行目)。本論文では、閾値選択アルゴリズムは[6]から引用している。上記の $t_{M}$ の定義から、時系列の超過分は次のように定義される。\[
Y_{t_{M}}:=\left(x_{i}-t_{M} \mid x_{i}>t_{M}\right) \quad \text { for } \quad x_{i} \in X .
\]

次に、サンプルがIDであるという仮定に対して最も敏感と考えられる候補点の初期順序集合 $C$ を決定する。 $C$ の定義は、最初の候補点を定義する基準として使用される入力パラメータ $p_{c_{1}}$ に依存する。\[
c_{1}:=\operatorname{Quantile}\left(X, p_{c_{1}}\right),
\]
ここで、 $p_{c_{1}}>p_{M}$ である。 $p_{M}$ と $p_{c_{1}}$ の両方の選択については、5.1節で説明する。したがって、候補の順序付き集合 $C$ は次のように初期化される (4行目)。
\[
C:=\left(x_{i} \in X \mid x_{i} \geq c_{1}\right)
\]

 $C$ の各候補について、アルゴリズムはID仮説に有意な影響を与えるかどうかを検証する。そのためには、 $X$ から $C$ の要素を除いた集合、すなわち $X^{0}:=X \backslash C$  (5行目)を考慮する必要がある。そして、 $X^{0}$  (6行目)における過剰サンプルは以下のように計算される。
\[
Y_{t_{M}}^{0}:=\left(x_{i}-t_{M} \mid x_{i}>t_{M}\right) \quad \text { for } \quad x_{i} \in X^{0} \quad (26)
\]

次に、サブセット $Y_{t_{M}}^{0}$ を用いて (7行目)、サンプルの参照EVIパラメータ ( $\xi_{t_{M}}^{0}$ で示される)を推定する。このステップは、TailIDの中核原理を理解する上で非常に重要である。 $Y_{t_{M}}^{0}$ の値は分布の裾野部分であり、最も極端な値の一部が除去された後、後で使用するためにセット $C$ に集められる。

初期化フェーズ ( $\underline{\text { line 8. }}$ ) の最後のステップとして、TailID はこの EVI ( $I_{t_{M}}^{0}$ ) の信頼区間を信頼水準 $\gamma$ で計算する。TailID は、最大極値の分布の真のパラメータ値が確率 $\gamma$ で含まれると予想される範囲を推定し、裾の挙動を特徴付ける。裾に混合がない場合、 $C$ に格納されている極値を追加した EVI の再推定は、依然として信頼区間内に収まるという議論がある。信頼区間は、アルゴリズムのコアブロックの各反復で再計算され、 $S$ に追加されない $C$ の各極値によってもたらされる追加情報が考慮される。 $\gamma$ の選択と推奨値の意味については、サブセクション 5.1 で説明する。
アルゴリズムの中核は、各要素 $c_{i} \in C$  ( $\underline{\text { line } \mathbf{9}}$ )に適用され、これらは11行目でサンプル $X^{1}:=X^{0} \cup c_{1}$ として追加される。 $C$ は順序付き集合であるため、最初の候補値 $c_{1}$ はサンプル平均値に最も近い値となり、順序付き集合 $C$ の最初の点である $c_{1}$ となる。その後、超過分 $Y^{i}$ は、新たに拡張されたサンプル集合 $X^{i}(\underline{\text { line 12 }})$ に対して、式26に従って計算される。

新たに計算された超過集合 $Y_{t_{M}}^{i}$ は、EVI $\xi_{t_{M}}^{i}$  (13行目)をフィッティングするために使用される。 $\xi_{t_{M}}^{i}$ が信頼区間 $I_{t_{M}}^{i-1}$  (14行目)内に収まらない場合、
\[
\xi_{t_{M}}^{i}\left(Y_{t_{M}}^{i}\right) \notin I_{t_{M}}^{i-1}
\]
 $c_{i}$ は、時系列の末尾がIDであるという仮説に敏感であると結論付けるのに十分な証明がある。したがって、 $c_{i}$ はID敏感点集合 $S$  ( $\underline{\text { line 15 }}$ )に追加される。ただし、 $\xi_{t_{M}}^{i}$ が信頼区間 $I_{t_{M}}^{i-1}$ 内にある場合は、 $S$ には追加されず、 $\xi_{t_{M}}^{i}$ は次の候補値 $c_{i+1}(\underline{\mathbf{l n e 1 7}})$ の信頼区間の更新に使用される。

アルゴリズムは、候補の中で最初のIDセンシティブポイント $c_{k}$ が見つかるまで、 $C$ 内の全ての要素を反復的に分析し、同じ手順を繰り返す。その時点で、 $C$ は順序付き集合であるため、後続の候補である $\left\{c_{j} \in C \mid j \geq k\right\}$ も全てセンシティブとなり、 $S$ に追加される (20行目)。この後者の動作は、最初のセンシティブポイントが見つかるまで (10行目)、 $S$ が空であるという仮定の下で強制される。

最後に、アルゴリズムは、ID 仮定を破る感度ポイントのリストとして $S$ を返す (23 行目)。

理論的な観点から見ると、TailID は一般的な一連の仮定に基づいていることを考慮する価値がある。第一に、データの裾がGPDに従うと仮定する。第二に、式18の信頼区間の妥当性には、EVI $\xi$ のMLEが漸近正規性を持つことが必要である。これは、正則性条件に依存する。具体的には、データ $X$ は $\operatorname{PDF} f\left(\xi_{0} \mid x_{i}\right)$ とIIDであり、 $\xi \neq \xi_{0}$ に対して $f\left(\xi \mid x_{i}\right) \neq f\left(\xi_{0} \mid x_{i}\right)$ が成り立ち、パラメータ空間 $\mathcal{X}$ はコンパクトであり、真のパラメータ $\xi_{0}$ はその内部に存在すると仮定する。さらに、 $\log f\left(\xi \mid x_{i}\right)$ は確率1で $\xi$ および $\mathbb{E}\left[\sup _{\xi \in \mathcal{X}}\left|\log f\left(\xi \mid x_{i}\right)\right|\right]<\infty$ において連続である。さらに、関数 $f\left(\xi \mid x_{i}\right)$ は2回連続微分可能であり、 $\xi_{0}$ の近傍 $\mathcal{N}$ において正に正であると仮定する。以下の積分可能性条件を満たす必要がある。 $\int \sup _{\xi \in \mathcal{N}}\left\|\nabla_{\xi} f\left(\xi \mid x_{i}\right)\right\| d X<\infty$ 、 $\int \sup _{\xi \in \mathcal{N}}\left\|\nabla_{\xi \xi} f\left(\xi \mid x_{i}\right)\right\| d X<\infty$ も満たされなければならない。さらに、フィッシャー情報行列が存在し、かつ特異でないことが要求され、 $\mathbb{E}\left[\sup _{\xi \in \mathcal{N}}\left\|\nabla_{\xi \xi} \log f\left(\xi \mid x_{i}\right)\right\|\right]<\infty$ [36] も満たされなければならない。
