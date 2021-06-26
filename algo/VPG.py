from .Agent import RL_agent


class VPG_agent(RL_agent):
    
    def __init__(self, 
                 policy_net, 
                 value_net,
                 **kwargs):

        super(VPG_agent, self).__init__(
            policy_net,
            value_net,
            **kwargs,
            algo_name='VPG'
        )
    
