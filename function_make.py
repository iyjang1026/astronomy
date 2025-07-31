"""
STEP 16
복잡한 계산 그래프(구현 편)
"""
import numpy as np
import cpnn.functions as F

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{}은(는) 지원하지 않습니다.".format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0    # 추가

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []    # 추가
        seen_set = set()    # 추가

        def add_func(f):    # 추가
            if f not in seen_set:    # 추가
                funcs.append(f)    # 추가
                seen_set.add(f)    # 추가
                funcs.sort(key=lambda x: x.generation)    # 추가

        add_func(self.creator)    # 추가

        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            for x, gx in zip(f.inputs, gxs): #grad를 input에 matching
                if x.grad is None:    # 변경
                    x.grad = gx    # 변경
                else:    # 변경
                    x.grad = x.grad + gx    # 변경

                if x.creator is not None: #true : creator를 더 넣음, false : while문 종료
                    add_func(x.creator)

    def cleargrad(self):
        self.grad = None


class Function():
    def __call__(self, *inputs): #call이 원래 없는데 정의를 해줌
        xs = [x.data for x in inputs] #ndarray들의 리스트로 만들어짐
        ys = self.forward(*xs)
        if not isinstance(ys, tuple): #ys가 튜플인지 검사
            ys = (ys, ) #튜플이 아니면 튜플로 만들기
        outputs = [Variable(as_array(y)) for y in ys] #

        self.generation = max([x.generation for x in inputs])    # 추가
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy

class Log(Function):
    def forward(self, x):
        y = np.log10(x)
        return y
    
    def backward(self, gy):
        gx = (1/x)*gy
        return gx

def log(x):
    return Log()(x)

def add(x0, x1):
    return Add()(x0, x1)


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def square(x):
    return Square()(x)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


if __name__ == "__main__":
    import gc

    print(gc.collect())
    for i in range(10):
        x = Variable(np.random.randn(10000))
        b = 1
        c = 1
        a = 1
        y = F.log(x)
        y = F.linear(y,a,c)
        print(y)
    print("GC 실행 후, 메모리에서 해제됨", gc.collect())
