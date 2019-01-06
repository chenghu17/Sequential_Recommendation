## Sequence Models

### Popularity Method
recommend items based on the number of occurrences

### BPR_MatrixFactorization
Bayesian Personalized Ranking

### FPMC
use Markov Chain、personalized recommendation

### HRM
use max-pooling、mean-pooling to generate session representation

### GRU4Rec
use GRU for every session. There is no connection between sessions.

### NARM
use global encoder and local encoder for single session.

### HGRU
use user embedding、GRU for every session. There is a connection between each person's session.


### SHAN
use pooling and attention to model all users' record

### STAMP
use pooling、attention、MLP to model single session

### SR-GNN
build graph、use GNN

### reference
1、Rendle, Steffen , et al. "BPR: Bayesian Personalized Ranking from Implicit Feedback." Conference on Uncertainty in Artificial Intelligence AUAI Press, 2009.

2、Rendle, S., Freudenthaler, C., & Schmidt-Thieme, L. (2010). Factorizing personalized Markov chains for next-basket recommendation. Proceedings of the 19th International Conference on World Wide Web.

3、Wang, Pengfei, et al. "Learning Hierarchical Representation Model for NextBasket Recommendation." (2015):403-412.

4、Session-based Recommendations with Recurrent Neural Networks,2016,ICLR.

5、Jing Li, Pengjie Ren, Zhumin Chen, Zhaochun Ren, Tao Lian and Jun Ma (2017). Neural Attentive Session-based Recommendation. In Proceedings of CIKM'17, Singapore, Singapore, Nov 06-10, 2017.

6、HGRU2Rec: Personalizing Session-based Recommendation with Hierarchical Recurrent Neural Networks,2017,RecSys.

7、Sequential Recommender System based on Hierarchical Attention Network,2018,IJCAI.

8、STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation,2018,KDD.

9、SR-GNN: Session-based Recommendation with Graph Neural Networks,2019,AAAI.
