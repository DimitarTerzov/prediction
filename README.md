## Data Model

```bash
Output for  sp_help vwSeedionData;

Column_name	Type	Computed	Length	Prec	Scale	Nullable	TrimTrailingBlanks	FixedLenNullInSource	Collation
PlayerId	uniqueidentifier	no	16	     	     	no	(n/a)	(n/a)	NULL
PlayerFirstName	nvarchar	no	200	     	     	no	(n/a)	(n/a)	Latin1_General_CI_AI
PlayerLastName	nvarchar	no	200	     	     	no	(n/a)	(n/a)	Latin1_General_CI_AI
PlayerDateOfBirth	varchar	no	10	     	     	yes	no	yes	SQL_Latin1_General_CP1_CI_AS
PlayerGender	varchar	no	6	     	     	no	no	no	SQL_Latin1_General_CP1_CI_AS
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


## Baseline Heuristic

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
