

class A:
    def __init__(self):
        self.str_ = 'Class A init'
        print('This is Class A.')

    def push(self):
        print('Run Class A push function.')

    @classmethod
    def test_classmethd(A):
        print('This is a class_method in Class A')

    @staticmethod
    def test_staticmethod():
        print('This is a static_method in Class A')

    def reurn_val(self):
        return self.str_

class B(A):
    def __init__(self):
        super(B, self).__init__()

    def push(self):
        print('Overwrite push function in Class A.')

    def call_classA_classmethod(self):
        self.test_classmethd()
        self.test_staticmethod()

    def reurn_val(self):
        return super(B, self).reurn_val()

if __name__ == '__main__':
    B().push()
    B().call_classA_classmethod()

    temp = B().reurn_val()
    print(temp)