import os

if os.path.exists('../output') == False:
    os.mkdir(os.getcwd()[:-3]+'output')
    os.mkdir(os.getcwd()[:-3]+'output/ex1')
    os.mkdir(os.getcwd()[:-3]+'output/ex2')
    os.mkdir(os.getcwd()[:-3]+'output/ex3')

