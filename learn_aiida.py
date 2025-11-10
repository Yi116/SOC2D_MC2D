import aiida as ai
from aiida.engine import calcfunction, workfunction, WorkChain
from aiida.orm import Float



@calcfunction
def add(x , y) :
    return x + y


@calcfunction 
def mult(x,y) :
    return x*y


@workfunction ### workflow, can call the calcfunction
def add_and_mult(x,y,z):
    sum = add(x,y)
    product = sum * z
    return product, product.store()


### workchain, save the progress between steps in workfunction with class workchain
# class addqdf(WorkChain):
 