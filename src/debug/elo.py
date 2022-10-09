import numpy as np

def expected_score(rating1, rating2):
    return 1 / (1 + 10 ** ((rating2 - rating1) / 400))

def update_ratings(elo1, elo2, num_games, num_wins1, K=30):
    num_wins2 = num_games - num_wins1
    score1 = expected_score(elo1, elo2)
    score2 = expected_score(elo2, elo1)
    elo1 += K * (num_wins1 / num_games - score1)
    elo2 += K * (num_wins2 / num_games - score2)
    return elo1, elo2