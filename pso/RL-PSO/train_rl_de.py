import gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from env import PSO_SHADE_Env
# ================== 训练配置 ==================
def train_dqn():
    # 创建并行环境
    env = make_vec_env(
        lambda: PSO_SHADE_Env(
            dim=10,
            max_iter=1000,
            num_function=10,  # 10个测试函数
            start_function_id=0
        ),
        n_envs=4  # 并行环境数
    )

    # DQN超参数配置
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=5e-4,
        buffer_size=200000,
        batch_size=256,
        exploration_fraction=0.3,
        exploration_final_eps=0.01,
        target_update_interval=500,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        seed=42
    )

    # 评估回调
    eval_callback = EvalCallback(
        env,
        best_model_save_path="./logs/",
        log_path="./logs/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    # 训练模型
    model.learn(
        total_timesteps=2e5,
        callback=eval_callback,
        progress_bar=True
    )
    
    # 保存模型
    model.save("dqn_pso_shade")
    return model

# ================== 测试代码 ==================
def evaluate_strategy(strategy, func_ids, num_episodes=10):
    """评估指定策略在不同函数上的表现"""
    results = {}
    for fid in func_ids:
        env = PSO_SHADE_Env(
            dim=10,
            max_iter=1000,
            start_function_id=fid,
            num_function=1
        )
        fitness_history = []
        
        for _ in range(num_episodes):
            obs = env.reset()
            episode_fitness = []
            done = False
            
            while not done:
                if strategy == "random":
                    action = env.action_space.sample()
                elif strategy == "trained":
                    action, _ = model.predict(obs, deterministic=True)
                else:  # 固定策略
                    action = strategy
                    
                obs, _, done, _ = env.step(action)
                episode_fitness.append(env.gbest_fitness)
            
            fitness_history.append(np.min(episode_fitness))
        
        results[f"f{fid+1}"] = {
            "mean": np.mean(fitness_history),
            "std": np.std(fitness_history)
        }
    return results

# ================== 主程序 ==================
if __name__ == "__main__":
    # 训练模型
    model = train_dqn()
    
    # 加载预训练模型
    model = DQN.load("dqn_pso_shade")

    # 测试配置
    test_functions = list(range(10))  # 测试10个函数
    strategies = {
        "Random": "random",
        "Exploration (0000)": 0,
        "Exploitation (1111)": 15,
        "Trained DQN": "trained"
    }

    # 运行测试
    results = {}
    for name, strategy in strategies.items():
        print(f"Testing {name} strategy...")
        results[name] = evaluate_strategy(strategy, test_functions)

    # 打印结果
    print("\n=== 性能比较（均值±标准差） ===")
    for func in [f"f{i+1}" for i in test_functions]:
        print(f"\nFunction {func}:")
        for strategy in strategies:
            data = results[strategy][func]
            print(f"{strategy:<15} | {data['mean']:.2e} ± {data['std']:.2e}")

    # 可视化（示例）
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    for strategy in strategies:
        means = [results[strategy][f"f{i+1}"]["mean"] for i in test_functions]
        plt.plot(means, 'o-', label=strategy)
    plt.xlabel("Function ID")
    plt.ylabel("Best Fitness (log)")
    plt.yscale("log")
    plt.title("Performance Comparison on CEC2017 Benchmark")
    plt.legend()
    plt.grid(True)
    plt.show()