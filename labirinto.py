# %% [markdown]
# # TD Learning: SARSA e *Q*-learning
# 
# #### Prof. Armando Alves Neto - Introdução ao Aprendizado por Reforço - PPGEE/UFMG
# 
# Objetivo: ensinar um robô a navegar através de um labirinto até um ponto-alvo específico.
# 
# <img src="problema_labirinto.png" width="300">
# 
# ## Características do labirinto:
# 
# ### Espaço de observações
# 
# O labirinto corresponde a um espaço de 10x10 metros, discretizado em um grid de 25x25.
# 
# ### Espaço de ações
# 
# O robô pode dar um passo em todas as 8 direções (todos os vizinhos são alcançáveis), ou pode ficar parado.
# 
# ### Função de recompensa
# 
# - Se alcançar o objetivo, recebe +100
# - Se o número de passo exceder 100, recebe -20
# - Se o robô colidir com algum obstáculo, recebe -50

# %%
import class_maze as cm
import numpy as np
from functools import partial
import os
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8,4)

# %% [markdown]
# Criando a classe para o TD learning.

# %%
class TDlearning(object):
    def __init__(self, parameters):

        self.parameters = parameters

        # metodo
        self.method = parameters['method']

        # numero de episodios
        self.episode = 0

        # cria o ambiente
        self.env = cm.Maze(render=parameters['render'])
        self.env.load_known_map()

        # tamanho dos espacos de estados e acoes
        self.num_states = np.prod(np.array(self.env.num_states))
        self.num_actions = self.env.action_space.n

        # parametros de aprendizado
        self.gamma = parameters['gamma']
        self.eps = parameters['eps']
        self.alpha = parameters['alpha']

        # métricas de exploração
        self.exploration_points = [25, 50, 75, 100]
        self.exploration_events = {}
        self.current_exploration = 0.0

        # log file (nome depende do metodo)
        self.logfile = parameters['q-file']
        if self.method == 'Q-learning':
            self.logfile = 'qlearning_' + self.logfile
        else: print("Não salvou...")

        # reseta a politica
        self.reset()

    ##########################################
    # reseta a funcao acao-valor
    def reset(self):
        # reseta o ambiente
        S = self.env.reset()
        
        # Q(s,a)
        self.Q = np.zeros((self.num_states, self.num_actions))

        # carrega tabela pre-computada se for o caso
        if self.parameters['load_Q']:
            try:
                with open(self.logfile, 'rb') as f:
                    data = np.load(f)
                    self.Q = data['Q']
                    self.episode = data['episodes']
            except: None

    ##########################################
    # retorna a politica corrente
    def curr_policy(self, copy=False):
        if copy:
            return partial(self.TabularEpsilonSoftPolicy, np.copy(self.Q))
        else:
            return partial(self.TabularEpsilonSoftPolicy, self.Q)
        
    ########################################
    # salva tabela Q(s,a)
    def save(self):
        with open(self.logfile, 'wb') as f:
            np.savez(f, Q=self.Q, episodes=self.episode)

    def check_exploration_points(self):
        explored = self.env.get_percentage_explored(only_free=True)
        self.current_exploration = explored

        for points in self.exploration_points:
            if points not in self.exploration_events and explored >= points:
                self.exploration_events[points] = {
                    'episode': self.episode,
                    'percentage': explored
                }

                print(
                    f"Porcentagem de exploração atingido: {points}% "
                    f"| episódio {self.episode} ",
                    flush=True
                )

        return explored

    ##########################################
    def __del__(self):
        self.env.close()

# %% [markdown]
# Probabilidade de escolha de uma ação $a$ baseada na política $\varepsilon$-soft:
# $$
# \pi(a|S_t) \gets 
#                         \begin{cases}
#                             1 - \varepsilon + \varepsilon/|\mathcal{A}|,  & \text{se}~ a = \arg\max\limits_{a} Q(S_t,a),\\
#                             \varepsilon/|\mathcal{A}|, & \text{caso contrário.}
#                         \end{cases}
# $$

# %%
class TDlearning(TDlearning):
    ##########################################
    # escolha da açao (epsilon-soft)
    def TabularEpsilonSoftPolicy(self, Q, state):
        # probabilidades de escolher as acoes
        p1 = 1.0 - self.eps + self.eps/self.num_actions
        p2 = self.eps/self.num_actions
        prob = [p1 if a == Q[state, :].argmax() else p2 for a in range(self.num_actions)]
        
        return np.random.choice(self.num_actions, p=np.array(prob))

# %% [markdown]
# Método do SARSA:
# - aplique ação $A$, receba $S'$ e $R$
# - escolha $A'$ a partir de $S'$ usando $Q$ ($\varepsilon$-greedy, por exemplo)
# - $Q(S,A) \gets Q(S,A) + \alpha \big[R + \gamma Q(S',A') - Q(S,A)\big]$
# - $S \gets S'$
# - $A \gets A'$


    ##########################################
# %% [markdown]
# Método do *Q*-learning:
# - escolha $A$ a partir de $S$ usando $Q$ ($\varepsilon$-greedy, por exemplo)
# - aplique ação $A$, receba $S'$ e $R$
# - $Q(S,A) \gets Q(S,A) + \alpha \big[R + \gamma \max\limits_a Q(S',a) - Q(S,A)\big]$
# - $S \gets S'$

# %%
class TDlearning(TDlearning):
    ##########################################
    def qlearning(self, S):
        
        # \pi(s)
        A = self.policy(S)

        # passo de interacao com o ambiente
        [Sl, R, done, _] = self.env.step(A)
        
        self.Q[S, A] = self.Q[S, A] + self.alpha*(R + self.gamma*self.Q[Sl, :].max() - self.Q[S, A])
        
        return Sl, R, done

# %% [markdown]
# Executando um episódio.

# %%
class TDlearning(TDlearning):
    ##########################################
    def runEpisode(self, render_every=1):

        # novo episodio
        self.episode += 1

        # pega a politica corrente (on-policy)
        self.policy = self.curr_policy()
        
        # inicia o ambiente (começa aleatoriamente)
        S = self.env.reset()

        self.check_exploration_points()
        
        # \pi(s)
        A = self.policy(S)

        # lista de rewards
        rewards = []

        while True:
            # Q-learning
            if self.method == 'Q-learning':
                Sl, R, done = self.qlearning(S)
            else:
                print('Método errado.')
            
            # proximo estado
            S = Sl

            # Salva rewards
            rewards.append(R)

            self.check_exploration_points()

            # renderiza o ambiente
            if (self.episode-1)%render_every == 0:
                self.env.render(self.Q)
                self.env.render_known_map()

            # chegou a um estado terminal?
            if done: break
        
        # salva a tabela Q
        if self.parameters['save_Q']:
            self.save()
            self.env.save_known_map()

        success = self.env.reached_goal()
        return np.sum(np.array(rewards)), success, self.current_exploration


# %% [markdown]
# Código principal:
# - episodes: número de episódios
# - max_steps: número máximo de passos no episódio
# - gamma: fator de desconto
# - eps: $\varepsilon$
# - alpha: $\alpha$
# - method: *SARSA* ou *Q-learning*
# - render: renderiza animação
# - save_Q: salva tabela *Q*
# - load_Q: carrega tabela *Q*
# - q-file: arquivo da tabela *Q*

# %%
if __name__ == '__main__':
    
    plt.ion()

    # parametros
    parameters = {'episodes'  : 2000,
                  'gamma'     : 0.99,
                  'eps'       : 1.0e-1,
                  'alpha'     : 0.5,
                  'method'    : 'Q-learning', #'SARSA' ou 'Q-learning'
                  'render'    : True,
                  'save_Q'    : True,
                  'load_Q'    : False,
                  'q-file'    : 'qtable.npy',}

    # TD algorithm
    mc = TDlearning(parameters)

    # historico de recompensas
    rewards = []
    avg_rewards = []

    # taxa de sucesso
    successes = []
    success_rate = []
    exploration_history = []
    
    while mc.episode <= parameters['episodes']:
        # roda um episodio
        total_reward, success, percentage_explored = mc.runEpisode()
        
        # rewards
        rewards.append(total_reward)
        # reward medio
        avg_rewards.append(np.mean(rewards[-50:]))
        successes.append(1 if success else 0)
        success_rate.append(np.mean(successes[-50:]))
        exploration_history.append(percentage_explored)
        
        # plot rewards
        plt.figure(1)
        #clear_output(wait=True)
        #print("\033[H\033[J", end="")
        plt.clf()
        plt.plot(avg_rewards, 'b', linewidth=2)
        plt.plot(rewards, 'r', alpha=0.3)
        plt.title('Recompensa por episódios')
        plt.xlabel('Episódios')
        plt.ylabel('Recompensa')
        plt.show()
        plt.pause(.1)

    plt.figure(3)
    plt.clf()
    plt.plot(success_rate, 'g', linewidth=2)
    plt.title('Taxa de sucesso por episódios')
    plt.xlabel('Episódios')
    plt.ylabel('Taxa de sucesso (média móvel 50 episódios)')
    plt.ylim(-0.05, 1.05)
    plt.savefig("results/success_rate.png", dpi=300, bbox_inches='tight')

    plt.figure(1)
    plt.clf()
    plt.plot(avg_rewards, 'b', linewidth=2)
    plt.title('Recompensa por episódio')
    plt.xlabel('Episódios')
    plt.ylabel('Recompensa')
    plt.savefig("results/reward_final.png", dpi=300, bbox_inches='tight')

    final_success_rate = np.mean(successes[-100:]) * 100
    print(f"Taxa de sucesso nos últimos 100 episódios: {final_success_rate:.2f}%")


