import torch
import numpy as np
from NeuralNetworks import ValueNet,PolicyNet
from DataLoader import DataLoader
import matplotlib.pyplot as plt


class ActorCritic:
    def __init__(self,in_size=3,out_size=1,N=15,N_action=100,N_policy_epoch=1,K=5,N_value_iter=20,N_policy_iter=10,N_total_iter=2,save_index=1):
        self.valuenet=ValueNet(in_size=in_size,outsize=1)
        self.policynet=PolicyNet(in_size=in_size,outsize=out_size)

        self.N=N
        self.N_action=N_action
        self.N_policy_epoch=N_policy_epoch
        self.K=K
        self.batch_size=int(self.N/self.K)

        self.N_value_iter=N_value_iter
        self.N_policy_iter=N_policy_iter
        self.N_total_iter=N_total_iter

        self.save_index=save_index
        
        self.loader=DataLoader(N=self.N)
        

        self.optimizer_valuenet = torch.optim.Adam(self.valuenet.parameters(), lr=1e-4) 
        self.optimizer_policynet = torch.optim.Adam(self.policynet.parameters(), lr=1e-4) 

        self.loss_valuenet=torch.nn.MSELoss()
        self.loss_policynet=torch.nn.MSELoss()

        self.valuenet_path="DModels/valuenet_{}.pth".format(save_index)
        self.policynet_path="DModels/policynet_{}.pth".format(save_index)

        self.value_loss=[]
        self.policy_loss=[]

    def run(self):
        for m in range(self.N_total_iter):
            print("Running main iteration : {}".format(m))
            
            self.policynet.eval()
            states,next_states,rewards=self.loader.generate_data(policy=self.policynet)
            for k in range(self.N_value_iter):
                value_losses=[]
                for i in range(self.K):
                    init_index=int(i*self.batch_size)
                    final_index=int((i+1)*self.batch_size)
                    batch_states=states[init_index:final_index,:]

                    batch_next_states=next_states[init_index:final_index,:]
                    batch_rewards=rewards[init_index:final_index]
                    
                    value_loss=self.value_iteration(batch_states,batch_next_states,batch_rewards)
                    value_losses.append(value_loss)
                self.value_loss.append(np.mean(value_losses))
            print("Current value loss {}".format(self.value_loss[-1]))
            
            optimal_actions=self.MC(states=states)
            for k in range(self.N_policy_iter):
                policy_losses=[]
                
                for i in range(self.K):
                    
                    init_index=int(i*self.batch_size)
                    final_index=int((i+1)*self.batch_size)

                    batch_states=states[init_index:final_index,:]
                    batch_optimal_actions=optimal_actions[init_index:final_index].reshape((self.batch_size,1))
                    
                    policy_loss=self.policy_iteration(states=batch_states,optimal_actions=batch_optimal_actions)
                    
                    policy_losses.append(policy_loss)

                self.policy_loss.append(np.mean(policy_losses))
            print("Current policy loss {}".format(self.policy_loss[-1]))
            self.save_in_run(index=m+1)
            
    def value_iteration(self,states,next_states,rewards):
        states=torch.Tensor(states)
        next_states=torch.Tensor(next_states)

        self.valuenet.eval()
        next_values=self.valuenet(next_states)

        next_values=next_values.detach().numpy()

        rewards=np.array(rewards)

        target=torch.Tensor(rewards.reshape((self.batch_size,1))+next_values)

        self.optimizer_valuenet.zero_grad()
        self.valuenet.train()
        
        state_values=self.valuenet(states)

        value_loss=self.loss_valuenet(target,state_values)

        value_loss.backward()

        self.optimizer_valuenet.step()

        return value_loss.item()

    def policy_iteration(self,states,optimal_actions):
        self.valuenet.eval()

        """optimal_actions=np.zeros((self.batch_size,1))

        for i in range(self.batch_size):
            optimal_action=self.search_optimal_action(states[i,:])
            optimal_actions[i]=optimal_action"""

        optimal_actions=torch.Tensor(optimal_actions)

        self.policynet.train()
        states=torch.Tensor(states)

        for i in range(self.N_policy_epoch):
            self.optimizer_policynet.zero_grad()
            actions=self.policynet(states)
            loss_policy=self.loss_policynet(optimal_actions,actions)
            loss_policy.backward()
            self.optimizer_policynet.step()
        
        return loss_policy.item()
    

    def search_optimal_action(self,x):
        actions=np.random.uniform(-1,1,self.N_action)

        max_target=-1000000
        optimal_action=0
        for i in range(self.N_action):
            action=actions[i]
            next_state=self.loader.simple_model(x=x.reshape((3,1)),u=action)

            reward=self.loader.calculate_reward(x=next_state)
            
            next_state=torch.Tensor(next_state.reshape((1,3)))

            next_value=self.valuenet(next_state).detach().item()

            target=reward+next_value

            if target>max_target:
                max_target=target
                optimal_action=action
        
        return optimal_action
    
    def MC(self,states):
        optimal_actions=np.zeros((self.N,1))

        for i in range(self.N):
            state=states[i,:]
            action=self.search_optimal_action(x=state)
            optimal_actions[i]=action
        
        return optimal_actions

    def vis(self):

        fig, axs = plt.subplots(2)
        fig.suptitle('Vertically stacked subplots')
        axs[0].plot(self.value_loss,label="Value loss")
        axs[1].plot(self.policy_loss,label="Policy loss")


        plt.savefig("figs/train_loss_{}.png".format(self.save_index))
        np.savetxt("figs/value_loss_{}.txt".format(self.save_index),self.value_loss)
        np.savetxt("figs/policy_loss_{}.txt".format(self.save_index),self.policy_loss)


    def save(self):
        self.policynet.save(path=self.policynet_path)
        self.valuenet.save(path=self.valuenet_path)

    def save_in_run(self,index):
        self.valuenet_path="DModels/valuenet_{}_{}.pth".format(self.save_index,index)
        self.policynet_path="DModels/policynet_{}_{}.pth".format(self.save_index,index)
        self.policynet.save(path=self.policynet_path)
        self.valuenet.save(path=self.valuenet_path)





