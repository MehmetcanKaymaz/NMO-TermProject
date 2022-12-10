from ActorCritic import ActorCritic

actorcritic=ActorCritic(in_size=3,out_size=1,N=1000,N_action=100,N_policy_epoch=1,K=10,N_value_iter=50,N_policy_iter=10,N_total_iter=50)

actorcritic.run()
actorcritic.save()
actorcritic.vis()


