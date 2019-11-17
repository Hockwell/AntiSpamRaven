#ODC - OptimalDisjunctiveCombination
#OCC - OptimalConjunctivalCombination

#группа классов ищет наилучшую комбинацию алгоритмов ML.
#для поиска используется кастомная кросс-валидация и варианты объединения детектов
class AlgsBestCombinationSearcher(object):
    def __init__(self):
        self.combinations = []
        self.k_folds = []
        
    def prepare(self, X, y, k_folds_amount, algs): #Сочетания без повторений
        
        def generate_algs_combinations(): #для 4 слоёв, можно на базе рекурсии и дерева создать общий метод для n слоёв
            
            #требуется расставить эл-ты списка на 4 места,на первый взгляд необходимо найти сочетания без повторений,но
            #есть нюанс - требуется, чтобы элемент пустоты (None) повторялся, и только он, ибо необходимо проверить
            #алгоритм и по-одиночке, и в паре и т.д.
            
            def get_not_alg_element():
                n = len(self.algs)
                for i in range(n):
                    if self.algs[i][1] == None:
                        return (i, self.algs[i])
                    
            algs_amount = len(self.algs)
            #print(self.algs)
            not_alg_index,not_alg_element = get_not_alg_element()
            algs_indexes = list(range(0,algs_amount))
            algs_indexes = algs_indexes[0:not_alg_index] + algs_indexes[not_alg_index+1:] #добавляем None в конец
            algs_indexes.append(not_alg_element)
                  
            for i1 in range(0,algs_amount):
                j2 = i1+1 if i1+1 <= algs_amount-1 else algs_amount-1
                for i2 in range(j2, algs_amount):
                    j3 = i2+1 if i2+1 <= algs_amount-1 else algs_amount-1
                    for i3 in range(j3, algs_amount):
                        j4 = i2+1 if i3+1 <= algs_amount-1 else algs_amount-1
                        for i4 in range(j4 , algs_amount):
                            self.combinations.append((i1,i2,i3,i4))
                            
        def split_dataset_on_k_folds():
            #folds: k-1 - train, k-ый - valid
            #на выход список кортежей ((X_trainFolds, y_trainFolds), (X_validFold, y_validFold))
            def take_train_folds():
                lower_bound_train = upper_bound_valid if upper_bound_valid < samples_amount-1 else 0
                print (lower_bound_train, samples_amount)
                upper_bound_train = lower_bound_train + train_folds_size
                if (upper_bound_train > samples_amount-1):
                    upper_bound_train = upper_bound_train - (samples_amount-1)
                    X_part1, y_part1 = X[lower_bound_train: samples_amount], y[lower_bound_train: samples_amount] 
                else:
                    X_part1, y_part1 = [],[]
                    #part1 - до конца датасета, part2 -с начала датасета
                X_part2, y_part2 = X[:upper_bound_train], y[:upper_bound_train]
                return (X_part1+X_part2, y_part1 + y_part2)
                
            X_trainFolds = []
            X_validFold = []
            y_trainFolds = []
            y_validFold = []
            samples_amount = X.shape[0]
            valid_fold_size = int(samples_amount/self.k) #округление вниз
            train_folds_size = samples_amount - valid_fold_size
            #первая часть достаётся валидационному фолду, а остальные обучающим
            for i in range(self.k):
                lower_bound_valid = i*valid_fold_size
                upper_bound_valid = lower_bound_valid + valid_fold_size
                X_validFold, y_validFold = X[lower_bound_valid:upper_bound_valid], y[lower_bound_valid:upper_bound_valid]
                X_trainFolds, y_trainFolds = take_train_folds()
                self.k_folds.append((X_trainFolds, y_trainFolds), (X_validFold, y_validFold))
        
        self.k = k_folds_amount
        self.X = X
        self.y = y
        self.algs = algs #список алгоритмов с None - заглушка для "нет алгоритма", она обязательна, ибо
        #данные методы работают с индексами алгоритмов - их комбинируют, а уже по этим индексам будет осуществляться
        #расстановка алгоритмов по комбинациям, у пустоты тоже должен быть свой индекс
        generate_algs_combinations()
        print(self.combinations)
        #split_dataset_on_k_folds()
        
    def run_ODCSearcher(self):
        pass
       
    def run_OCCSearcher(self):
        #будет реализован, если понадобится
        pass

