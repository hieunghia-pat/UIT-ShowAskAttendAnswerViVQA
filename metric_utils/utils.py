def batch_accuracy(predicted, true):
    """ Compute the accuracies for a batch of predictions and answers """
    _, predicted_index = predicted.max(dim=1, keepdim=True)
    agreeing = true.gather(dim=1, index=predicted_index)
    '''
    Acc needs to be averaged over all 10 choose 9 subsets of human answers.
    While we could just use a loop, surely this can be done more efficiently (and indeed, it can).
    There are two cases for the 1 chosen answer to be discarded:
    (1) the discarded answer is not the predicted answer => acc stays the same
    (2) the discarded answer is the predicted answer => we have to subtract 1 from the number of agreeing answers
    
    There are (10 - num_agreeing_answers) of case 1 and num_agreeing_answers of case 2, thus
    acc = ((10 - agreeing) * min( agreeing      / 3, 1)
           +     agreeing  * min((agreeing - 1) / 3, 1)) / 10
    
    Let's do some more simplification:
    if num_agreeing_answers == 0:
        acc = 0  since the case 1 min term becomes 0 and case 2 weighting term is 0
    if num_agreeing_answers >= 4:
        acc = 1  since the min term in both cases is always 1
    The only cases left are for 1, 2, and 3 agreeing answers.
    In all of those cases, (agreeing - 1) / 3  <  agreeing / 3  <=  1, so we can get rid of all the mins.
    By moving num_agreeing_answers from both cases outside the sum we get:
        acc = agreeing * ((10 - agreeing) + (agreeing - 1)) / 3 / 10
    which we can simplify to:
        acc = agreeing * 0.3
    Finally, we can combine all cases together with:
        min(agreeing * 0.3, 1)
    '''
    return (agreeing * 0.3).clamp(max=1)