from ActorCritic import ActorCritic

actorcritic=ActorCritic(in_size=3,out_size=1,N=10000,N_action=100,N_policy_epoch=1,K=100,N_value_iter=5,N_policy_iter=2,N_total_iter=10,save_index=11)

actorcritic.run()
actorcritic.save()
actorcritic.vis()


