from ph import charge_isp
from hydrophobicty import gravy_score
from solubilty import solubility_score
from serine import check_sequence_all_aspartic


def reward_func(seq):
    values = [
        *charge_isp(seq),
        gravy_score(seq),
        solubility_score(seq),
        check_sequence_all_aspartic(seq)]
    return values
print(reward_func("AVKF"))
print(reward_func("AKTIAY"))
print(reward_func("KQRSRE"))


