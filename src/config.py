gamma   = 0.9
lambda_ = 0.9
alpha   = 0.1
epsilon = 0.01
eta     = 0.1

column_names = {
    'ptf1': {
        'quarterly':  {'spx': 'SPXret_1q', 'agg': 'AGGret_1q'},
        'semi_annual':{'spx': 'SPXret_s',  'agg': 'AGGret_s'},
        'annual':     {'spx': 'SPXret_a',  'agg': 'AGGret_a'}
    },
    'ptf3': {
        'quarterly':  {'spx': 'SPXret_1q', 'agg': 'TNXret_1q'},
        'semi_annual':{'spx': 'SPXret_s',  'agg': 'TNXret_s'},
        'annual':     {'spx': 'SPXret_a',  'agg': 'TNXret_a'}
    }
}