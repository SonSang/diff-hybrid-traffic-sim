import torch as th

def sigmoid(value: th.Tensor, 
            constant: th.Tensor, 
            min: th.Tensor = -16.0, 
            max: th.Tensor = 16.0):

    '''
    Sigmoid function that can be used for IF statements.

    @ value: The value that would be fed into the function.
    @ constant: If we already know the range of [value], then we can multiply
    some constant to make the output close to -1 and 1. Therefore, the final
    input to the sigmoid function would be [value * constant].
    @ min, max: The range of [value * constant] to prevent vanishing / exploding
    gradients.
    '''

    if not isinstance(value, th.Tensor):

        value = th.tensor(value)

    if not isinstance(constant, th.Tensor):

        constant = th.tensor(constant)

    value = value * constant
    value = th.clamp(value, min, max)
    
    return th.sigmoid(value)