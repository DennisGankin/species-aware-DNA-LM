#from src.datamodules.sequence_operations import mapping

mapping = "ACGTN"

def compare_sequences(pred, target):
    pred_string = "pred: "
    target_string = "true: "
    # indicator if we are at true or false
    true_streak = 0
    for p_base, t_base in zip(pred, target):

        if p_base == -100 or t_base == -100:
            continue

        # to nucleotide
        p_nuc = mapping[p_base]
        t_nuc = mapping[t_base]
        
        # compare
        if p_nuc == t_nuc:
            if true_streak == 0:
                # not yet set to blue
                pred_string += "\x1b[34m"
                target_string += "\x1b[34m"
            true_streak = 1

        if p_nuc != t_nuc:
            if true_streak == 1:
                # set back to black
                pred_string += "\x1b[0m"
                target_string += "\x1b[0m"
            true_streak = 0
            
        # add nucleotides
        pred_string += p_nuc
        target_string += t_nuc
        
    pred_string += "\x1b[0m"
    target_string += "\x1b[0m"
    return pred_string+ "\n"+ target_string




