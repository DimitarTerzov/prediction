# A Survey of Learning Models for Predicting the Outcome of Basketball Match-Ups

This README outlines the machine learning approach taken to
predict the winner of a 3x3 basketball match-up given historical
player data.

## Data Model

The following data was made available in an Azure MSSQL database instance:

```bash
sp_help vwSeedionData;

Column_name	Type	Computed	Length	Prec	Scale	Nullable	TrimTrailingBlanks	FixedLenNullInSource	Collation
PlayerId	uniqueidentifier	no	16	     	     	no	(n/a)	(n/a)	NULL
PlayerFirstName	nvarchar	no	200	     	     	no	(n/a)	(n/a)	Latin1_General_CI_AI
PlayerLastName	nvarchar	no	200	     	     	no	(n/a)	(n/a)	Latin1_General_CI_AI
PlayerDateOfBirth	varchar	no	10	     	     	yes	no	yes	SQL_Latin1_General_CP1_CI_AS
PlayerGender	varchar	no	6	     	     	no	no	no	SQL_Latin1_General_CP1_CI_AS
PlayerRankingPoints	float	no	8	53   	NULL	no	(n/a)	(n/a)	NULL
TeamName	nvarchar	no	100	     	     	no	(n/a)	(n/a)	Latin1_General_CI_AS
EventName	nvarchar	no	400	     	     	no	(n/a)	(n/a)	Latin1_General_CI_AS
EventSeason	int	no	4	10   	0    	yes	(n/a)	(n/a)	NULL
EventStartDate	datetimeoffset	no	10	34   	7    	no	(n/a)	(n/a)	NULL
EventEndDate	datetimeoffset	no	10	34   	7    	no	(n/a)	(n/a)	NULL
DivisionName	nvarchar	no	800	     	     	yes	(n/a)	(n/a)	Latin1_General_CI_AS
DivisionGender	varchar	no	6	     	     	no	no	no	SQL_Latin1_General_CP1_CI_AS
DivisionClassification	varchar	no	100	     	     	yes	no	yes	SQL_Latin1_General_CP1_CI_AS
DivisionCoreLevel	varchar	no	200	     	     	yes	no	yes	Latin1_General_CI_AS
GameName	nvarchar	no	2000	     	     	no	(n/a)	(n/a)	Latin1_General_CI_AS
GamePlayedAsHomeTeam	int	no	4	10   	0    	no	(n/a)	(n/a)	NULL
GameWonBy	varchar	no	4	     	     	no	no	no	SQL_Latin1_General_CP1_CI_AS
GameHomeTeamPoints	tinyint	no	1	3    	0    	yes	(n/a)	(n/a)	NULL
GameAwayTeamPoints	tinyint	no	1	3    	0    	yes	(n/a)	(n/a)	NULL
GameIsForfeit	bit	no	1	     	     	no	(n/a)	(n/a)	NULL
TeamFinalStanding	tinyint	no	1	3    	0    	no	(n/a)	(n/a)	NULL
```

Some basic statistics on the data:
* 112,061 teams
* 369,324 players
* 161,162 games

## Baseline Heuristic

### Heuristic 1: Aggregate Total Wins Per Player
The baseline heuristic is as follows:
- Every player gets a point for every game they have played where the
  team they were on won
- Every player loses a point for every game they have played where the
  team they were on lost

For a given match:
- Take all of the points for each member of a given team,
  and sum them together
- The team with the most points wins

We implement leave-one-out cross-validation in this heuristic by
simply subtracting the score a given player has for the game being
evaluated.

The code for this test lives in `heuristic.py`.

Here's an example output of that test:

```
Getting data...
Got data, took 198.35 seconds

Precision	0.30
Recall	0.30
True Neg	0.30
Acc	0.30
Fscore	0.30
```

Interestingly, what this means for the data set is that betting
*against* this heuristic gives you 70% accuracy in predicting the
winning team of a match.

### Heuristic 2: Aggregate Total Ranking Per Player

## Using Machine Learning

Determining whether a team will win or lose when playing against
another team can be considered a form of binary classification.
For each match-up, we generate two instances to learn or test against.
*Instance A* will be the first team, and *Instance B* will be the second team.
We will apply a class of 1 if a given instance denotes the team that
wins the match-up, and 0 if the instance denotes the team that loses.

The goal of a successful classifier is to therefore determine
whether a team will win a match-up given the other team.
The classifier will use features extracted from each match-up to
train each learning model.

### Base features

The following features are defined as a baseline set:

* Name of Team
* Name of Opposing Team
* Players on Team
* Players on Opposing Team
* Whether the Game was a Home Game
* Division Name
* Gender of Division

Features are initially expressed as binary values.
In instances where there are multiple values, the feature is transformed
as in the following pseudocode example:

```
Players = ['joe', 'mike', 'bob']
Team = 'Team 1'
Opposing Team = 'Team 2'
Features = {
  'team_Team 1': 1,
  'opposing_team_Team 2': 1,
  'player_joe': 1,
  'player_bob': 1,
  'player_mike': 1
}
```

When a feature set is calculated from this, we introduce a highly
dimensional feature set. This would generate a sparse matrix with a
size too large to fit into memory for most machines. To accommodate
this high dimensionality, we use the
[hashing trick](https://en.wikipedia.org/wiki/Feature_hashing)
to dynamically compute the representation of a feature during learning
to reduce memory and improve learning performance.

### Models Evaluated

Models are evaluated using
[ten-fold cross-validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)#k-fold_cross-validation).
This evaluates performance by training on 90% of the data,
and testing on the withheld 10%.
This is done ten times, all with the same folds, alternating the training
sets and testing set each time.

#### Stochastic Gradient Descent (SGD) Classifier with RBF Feature Transform (with Monte Carlo methods)

The [SGD Classifier](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
is a linear model that is suitable for scaling to large data sets with a large
number of features.

For this classifier, we apply kernel approximation to our hashed features
using the [Radial Basis Function Kernel](http://scikit-learn.org/stable/modules/kernel_approximation.html#rbf-kernel-approx).
This uses Monte Carlo methods to transform feature data in a way that
improves the speed of learning with minimal cost to the reliability of
the features.

#### Random Forest

Random Forest is a kind of
[ensemble learning](https://en.wikipedia.org/wiki/Ensemble_learning) approach
that builds out numerous randomized decision trees.

Ensemble learning is generally successful by smoothing a diverse
range of hypothesis against a data set into a well-rounded solution.

#### Naive Bayes

[Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
is a simple classifier that uses
prior probabilities of observed features for a given outcome.
It applies the joint probabilities of all observed features to
predict the class for a given instance.

While relatively unsophisticated, it provides a solid "bang for the buck"
in terms of speed and performance. However, it assumes that features
are independent, which is not necessarily the case in our scenario.

#### Logistic Regression

Logistic Regression, also known as Maximum Entropy, is a form
of [regression analysis](https://en.wikipedia.org/wiki/Regression_analysis)
that applies probabilistic learning to the problem of binary classification.

While logistic regression is similar to Naive Bayes in that is estimating
the probability of a class given a set of features, these probabilities
are calculated in such a way that is robust against features that
correlate, or are otherwise not necessarily independent.

#### Support Vector Machines (SVM)

Support Vector Machines are a kind of linear model that attempt to
separates a class definition from another class definition --
in this case, a "win" vs a "loss" --
by defining a hypothesis as a hyperplane in N-dimenstional space.

The dimensions of the space are generally an enumeration of
the set of the features, with each acting as a dimension in this space.

SVMs are another tool that are popular in spaces with
significant dimensionality.

#### Neural Network

Neural networks are a class of predictors that can use non-linear
functions to predict the outcome of a specific event. These learners
are popular in applications such as computer vision. They are designed
to automatically infer rules that inform the outcome for a given
set of features. The specific implementation we tested was a
Multi-Layered Perceptron Classifier (MLP).

### Results

| Method | Fit Time | Score Time | Precision | Recall | F-Score |
| ------ | --------:| ----------:| ---------:|-------:|--------:|
| Heuristic | N/A   | N/A        | 30        | 30     |   30    |
| Heuristic^-1 | N/A  | N/A      | 70        | 70     | 70      |
| SGD    | 3.3      | 0.12       | 54.94     | 44.0   | 48.32   |
| Random Forest | 16203.3 | 284.99  | 49.27  | 39.4   | 43.22   |
| Naive Bayes | 6.2  | 0.68      | 35.72     | 87.11  | 50.4    |
| Logistic Regression | 0.49 | 0.50 | 57.85 | 42.78 | 48.83 |
| SVM | 1267.48 | 0.47 | 46.53 | 54.1 | 49.6 |
| Neural Network | 529.83 | 0.37 | 47.4 | 69.6 | 56.3 |

*Times are in seconds. Values are averages over ten-fold cross validation.
 Precision, recall, and f-score are expressed as percentages.*


### Conclusion and Future Work
It appears that given the source data, the inverse approach of our
baseline heuristic remains unreasonably effective with an F-Score of 70%.

The best-performing machine learning model is the Naive Bayes implementation,
which performs roughly as well as a coin flip.

Some research into
[existing work in the field of machine learning for
basketball](https://www.researchgate.net/publication/257749099_Predicting_college_basketball_match_outcomes_using_machine_learning_techniques_some_results_and_lessons_learned)
has some observations that I believe may provide suggestions for
improving the outcomes of the learning models. These observations
include that using seasonal data to improve feature selection
may provide some performance improvement, while the choice of a given
machine learning model may not necessarily have a strong impact.
Notable as well is that the state of the art in using machine
learning to predict winners for basketball has so far only been
roughly *75% effective*, which means a significant investment in
feature selection for an increasingly diminishing return.


## Replicating the Results

```bash
# from the server hosting the code
$ cd ~/seedion

# enters the pipenv virtual environment
$ pipenv shell

# calculates the heuristic and outputs a report
# takes about five minutes, mostly from pulling data from the DB
$ python3 heuristic.py

# runs learners using pickled data from DB to speed things up
# You can remove the WITH_PICKLE=1 to re-query from the DB
# You can specify a classifier using, for example, CLASSIFIER=NAIVE_BAYES
$ WITH_PICKLE=1 NUM_JOBS=8 python3 learner.py
```
