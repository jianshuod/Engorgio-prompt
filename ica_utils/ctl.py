

class Converge_check():
    def __init__(self) -> None:
        self.counter = 0
        self.new_state = None
    
    def __call__(self, new_trigger_tokens):
        if self.new_state is None: 
            self.new_state = new_trigger_tokens
            return False

        if check_the_same(self.new_state, new_trigger_tokens):
            self.counter += 1
        else:
            self.new_state = new_trigger_tokens
            self.counter = 0
        if self.counter >= 50: return True
        else: return False


def check_the_same(l1, l2):
    res = True
    dis_list = []
    for idx, (l1x, l2x) in enumerate(zip(l1, l2)):
        if l1x != l2x: 
            res = False
            dis_list.append(idx)
    return res