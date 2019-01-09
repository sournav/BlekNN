class node:
    def __init__(self,bias,func):
        self.bias = bias
        self.func = func
    def aggregator(self,inp, weights):
        num_inputs = len(inp)
        sum_out = 0
        for i in range(num_inputs):
            sum_out+= (inp[i]*weights[i])
        return sum_out + self.bias
    def neuron(self,agr_inp):
        return self.func(agr_inp)

    def proc_neuron(self,inp,weights):
        return self.neuron(self.aggregator(inp,weights))
    def update_bias(self,bias):
        self.bias = bias


def identity(x):
    y = x
    return y


class layer:
    def __init__(self,inp_or_not,size,func = None):
        self.type = inp_or_not
        self.bias_vect = []
        for i in range(size):
            self.bias_vect.append(0)
        if(inp_or_not == True):
            self.layer = self.create_input_layer(size)
        else:
            self.layer = self.create_hidden_layer(size,func)
        self.size = size
    def get_size(self):
        return self.size
    def run_layer(self,inp,weights = None):
        if(self.type == True):
            return self.run_input_layer(inp)
        else:
            return self.run_hidden_layer(inp,weights)
    
    def create_input_layer(self,num_inputs):
        ret = []
        for i in range(num_inputs):
            ret.append(node(self.bias_vect[i],identity))
        return ret
    def run_input_layer(self,inp):
        ret = []
        num_inputs = len(inp)
        for i in range(num_inputs):
            ret.append(self.layer[i].proc_neuron([inp[i]],[1]))
        return ret
    def create_hidden_layer(self,num_inputs,func):
        ret = []
        for i in range(num_inputs):
            self.bias_vect.append(1)
            ret.append(node(self.bias_vect[i],func))
        return ret
    def run_hidden_layer(self,inp,weights):
        ret = []
        
        for i in range(self.size):
            
            ret.append(self.layer[i].proc_neuron(inp,weights[i]))
        return ret
    def get_nodes(self):
        return self.layer
class network:
    outs = []
    pdivs = []
    def __init__(self,structure,func,err,tgt = None):
        self.step_size = 0.5
        self.structure = structure
        self.nodes = self.build_nodes(structure,func[0])
        self.weights = self.build_weights(self.nodes)
        self.err = err[0]
        self.derr = err[1]
        self.tgt = tgt
        self.error_vals = []
        self.output = []
        self.dfunc = func[1]
        self.iterations = 100
        self.min_error = 10000000
        self.saved_weights = self.weights
        
    def build_nodes(self,structure,func):
        num_layers = len(structure)
        ret = [layer(True,structure[0])]    
        for i in range(1,num_layers):
            ret.append(layer(False,structure[i],func))
        return ret
    def build_weights(self,nodes):
        weights = []
        num_layers = len(nodes)
        for i in range(num_layers-1):
            wgts_per_neur = nodes[i].get_size()
            layer_weights = []
            for layer_node in nodes[i+1].get_nodes():
                node_weights = []
                for j in range(wgts_per_neur):
                    node_weights.append(1)
                layer_weights.append(node_weights)
            weights.append(layer_weights)
        return weights
    def feed_fwd(self,inp):
        layer_input_buf = self.nodes[0].run_layer(inp)
        #print(inp)
        self.outs.append(inp)
        #for i in layer_input_buf:
         #   if i not in [0]:
                #print(i)
        num_layers = len(self.nodes)
        for i in range(num_layers-1):
            layer_input_buf = self.nodes[i+1].run_layer(layer_input_buf,self.weights[i])
            self.outs.append(layer_input_buf)
        for i in range(self.structure[num_layers -1]):
            self.error_vals.append(self.err(layer_input_buf[i],self.tgt[i]))
        self.output = layer_input_buf
        return self.output
    
    def error_back(self):
        num_layers = len(self.nodes)
        curr_pdiv = []
        ind = num_layers- 2
        buf1 = self.weights[ind]
        len1 = len(buf1)
        tot_err = 0
        for j in range(len1):
            tot_err+=self.err(self.output[j],self.tgt[j])
        if self.min_error > tot_err:
            self.min_error = tot_err
            self.saved_weights = self.weights
        print('err: ', tot_err)
        for j in range(len1):
            
            temp1 = self.derr(self.output[j],self.tgt[j])
            temp2 = self.dfunc(self.output[j])
            err1 = temp1*temp2
            curr_pdiv.append(err1)
            buf2 = buf1[j]
            len2 = len(buf2)
            for k in range(len2):
                wadj = self.outs[-2][k]*err1
                self.weights[ind][j][k] -= (self.step_size*wadj)
        self.pdivs.append(curr_pdiv)
       
        
            
        return self.weights
    def hidden_back(self):
        num_hidden = len(self.nodes)-1
        pertinent = num_hidden - 1
        step = 1
        gap = 1
        for i in range(pertinent):
            ind = pertinent-i
            
            buf1 = self.weights[ind-1]
            len1 = len(buf1)
            temp_outs = self.outs[ind+1]
            
            len3 = len(temp_outs)
            curr_div = []
            len5 = len(self.pdivs[i])
            pdiv_len = len(self.outs[ind])
            
            curr_div = []
            
            for k in range(pdiv_len):
                for l in range(len5):
                    if(l==(len5-1)):
                        pdiv_ind = len5-1
                    else:
                        pdiv_ind = (l*gap)%(len5-1)
                    
                    pert_pdiv = self.pdivs[i][pdiv_ind]
                    
                    pert_weight = self.weights[ind][l%step][k]
                    pert_node = self.dfunc(self.outs[ind][k])
                    err1 = pert_pdiv*pert_weight*pert_node
                    curr_div.append(err1)
                    step = len(self.outs[ind+1])
            gap*=step
            jump = int(len(curr_div)/len(self.outs[ind]))
            shift = 0
            for k in range(len(self.outs[ind])):
               for l in range(len(self.outs[ind-1])):
                   
                   for m in range(jump):
                       #if ind-1 == 0:
                           #print((self.outs))
                       self.weights[ind-1][k][l]-= (self.step_size * curr_div[m+shift*jump] * (self.outs[ind-1][l]))
                       self.nodes[ind-1].layer[l].bias-= (self.step_size * (curr_div[m+shift*jump]))
               shift+=1
            self.pdivs.append(curr_div)
    def configure_runtime(self,iterations=100,step_size = 0.5):
        self.step_size = step_size
        self.iterations = iterations
    def run(self,inputs,outputs):
        num_inputs = len(inputs)
        num_outputs = len(outputs)
        if(num_inputs != num_outputs):
            print('\033[91m' + 'ERROR: NUMBER OF INPUTS IS NOT EQUAL TO NUMBER OF OUTPUTS' + '\033[0m')
        for i in range(self.iterations):
            self.tgt = outputs[i%num_outputs]
            self.feed_fwd(inputs[i%num_inputs])
            
            self.error_back()
            
            self.hidden_back()
            self.weights = self.saved_weights
            
class funcs:
    def identity(self,x):
        y = x
        return y
    def sigmoid(x):
        e = 2.718
        return 1/(1+e**(-x))
    def dsigmoid(x):
        e = 2.718
        return (1-(1/(1+e**(-x))))*(1/(1+e**(-x)))
    def mserr(inp,tgt):
        return 0.5*(tgt-inp)**2
    def dmserr(inp,tgt):
        return (inp-tgt)
    def didentity(y):
        return 1
#net = network([2,784,1],[sigmoid,dsigmoid],[err,derr],[1])
#print(net.feed_fwd([0,0]))
#net.configure_runtime(iterations = 300,step_size = 0.1)
#net.error_back()
#net.hidden_back()
#net.run([[0.5,0.5],[0.1,0.5],[0.5,0.1],[0.1,0.1]],[[0.1],[0.9],[0.9],[0.1]])
#print(net.feed_fwd([1,3,2,1]))
#print(net.feed_fwd([0.5,0.5]))
#print(net.weights)
#
#net.error_back() 
#net.hidden_back()
#print(net.feed_fwd([1,2,3,4]))
#print(net.weights)
#print(build_weights(build_nodes([2,3],identity)))        
#print(feed_fwd([1,2,3],build_weights(build_nodes([3,2,1],identity)),build_nodes([3,2,1],identity)))
#layers = layer(False,3,identity)
#print(layers.run_layer([1,2,3],
#                       [[1,2,3],[1,2,3],[1,2,3]]))
