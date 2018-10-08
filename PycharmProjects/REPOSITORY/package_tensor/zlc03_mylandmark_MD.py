import tensorflow as tf
import numpy as np

def add_layer_in(inputs,in_size,out_size,activation_function=None):

    '''definite input layer

    :argument:Weights_in, biases_in
    :return:outputs
    '''

    global Weights_in,biases_in
    Weights_in = tf.Variable(tf.random_normal([in_size,out_size]))
    biases_in = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs, Weights_in)  + biases_in
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def add_layer_out(inputs,in_size,out_size,activation_function=None):

    '''definate output layer

    :argument:Weights_out, biases_out
    :return:outputs
    '''

    global Weights_out,biases_out
    Weights_out = tf.Variable(tf.random_normal([in_size,out_size]))
    biases_out = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs, Weights_out)  + biases_out
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

xs = tf.placeholder(tf.float32,[None,6])
ys = tf.placeholder(tf.float32,[None,1])

prein = add_layer_in(xs,6,10,activation_function=tf.nn.relu)
prediction = add_layer_out(prein,10,1,activation_function=None)

loss = tf.reduce_sum(tf.square(prediction - ys))
#loss = -tf.reduce_sum(ys*tf.log(prediction))

#train_step = tf.train.GradientDescentOptimizer(0.00000001).minimize(loss)
train_step = tf.train.AdamOptimizer(1.).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

filerecord = open('result.txt','w')

filein_ener = open('oneline_energies_train.txt','r')
filein_ra = open('oneline_structures_train.xyz','r')

energy_tensor =[]
ra_tensor = []

for num in range(800):

    '''definite a circle read lines of your data
    
    read oneline_energies_train.txt and oneline_structures_train.xyz   
    get 800 tensors of energy by input values
    get 800 tensors of structures by input values
    transform structures into [800,6]
    transform energies into [800,1]
    :argument:line_ener, line_ra, energy
    :return:energy_tensor
    '''
    line_ener = filein_ener.readline()
    line_ra = filein_ra.readline()
    
    energy = float(line_ener)
          
    list = line_ra.split(' ')
    #print(list)
    for word in list:        
        if (word==''):
            print('kongge')
        elif(word=='\n'):
            print('mowei')
        else:
            ra_tensor.append(float(word))

    energy_tensor.append(energy)

ra_tensor = np.reshape(ra_tensor,(800,6))
energy_tensor = np.reshape(energy_tensor,(800,1))
#print(energy_tensor)

filein_ener.close()
filein_ra.close()    

for i in range(1000000000):

    '''
    trian 1000000000 times, print outloss every 1000 times and record it in result.txt;
    record weights_in or biases_in in result.txt when outloss < 1000. and i%10000==0 
    :return:
    '''


    sess.run(train_step, feed_dict={xs:ra_tensor,ys:energy_tensor})
    if i%1000 ==0:
        outloss = sess.run(loss,feed_dict={xs:ra_tensor,ys:energy_tensor})
        print(outloss,i)
#        print(sess.run([Weights_in,biases_in]))
#        print(sess.run([Weights_out,biases_out]))
#        print(Weights_in,biases_in,Weights_out,biases_out)
        filerecord.write('%d %f \n' %(i,outloss))
        
        if(outloss < 1000. and i%10000==0):
            filerecord.write('Weights_in:\n')
            for x in sess.run(Weights_in):
                filerecord.write(str(x)+'\n')
            filerecord.write('biases_in:\n')
            for x in sess.run(biases_in):
                filerecord.write(str(x)+'\n')
            filerecord.write('Weights_out:\n')
            for x in sess.run(Weights_out):
                filerecord.write(str(x)+'\n')
            filerecord.write('biases_out:\n')
            for x in sess.run(biases_out):
                filerecord.write(str(x)+'\n')
            filerecord.write('\n')

        #filerecord.write("\n".join(" ".join(map(str,x)) for x in (a)))
#        np.savetxt('a.txt',sess.run(Weights_out),fmt='%10.5f')

print('\nThe end of train.')

filerecord.close()

