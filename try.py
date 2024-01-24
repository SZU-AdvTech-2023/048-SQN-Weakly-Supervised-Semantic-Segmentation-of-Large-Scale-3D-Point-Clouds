class Student:
    def __init__(self, a):
        self.name = a
        
    def printa(self):
        print(self.name)
   
class B(Student):
    def __init__(self, a):
        super().__init__(a)
        

    def printa(self):
        print(self.name)
        
    def rename(self):
        self.name = "小明"
        
    
one = Student("张三")        

one_1 = B("李华") 
one_1.rename()    
one_1.printa()

one_2 = B("李华")  
one_2.printa()

one.printa()
     