import numpy as np

class Particle:
    """
    Tạo 1 cá thể trong bầy gồm các tham số: hàm fitness, dim, max_x, min_x
    """
    def __init__(self, fitness, dim, max_x, min_x, seed = None):
        if seed is not None:
            np.random.seed(seed)

        self.rnd = np.random.rand()
        self.max_x = max_x
        self.min_x = min_x
        self.max_v = self.rnd * (max_x - min_x)
        self.min_v = -self.rnd * (max_x - min_x)

        self.position = np.random.uniform(min_x, max_x, dim)
        self.velocity = np.random.uniform(-self.rnd * (max_x - min_x), self.rnd * (max_x - min_x), dim)

        self.fitness = fitness(self.position)

        self.best_pos = self.position.copy() 
        self.best_fitness = self.fitness # Giá trị tốt nhất của cá thể
    
    def limit_x(self, X):
        """
        Kiểm tra vị trí cá thể có nằm trong bầy
        """
        for i in range(len(X)):
            if X[i] > self.max_x:
                X[i] = self.max_x
            if X[i] < self.min_x:
                X[i] = self.min_x
        return X
    
    def limit_v(self, V):
        """
        Kiểm tra vận tốc cá thể có nằm trong vận tốc bầy
        """
        for i in range(len(V)):
            if V[i] > self.max_v:
                V[i] = self.max_v
            if V[i] < self.min_v:
                V[i] = self.min_v
            
        return V

def pso(fitness, max_iter, n, dim, min_x, max_x, w = 0.95, c1 = 1.49445, c2 = 1.49445, minimize = True, seed = None):
    if seed is not None:
        np.random.seed(seed)
        swarm = [Particle(fitness, dim, max_x, min_x, seed + i) for i in range(n)]
    else:
        swarm = [Particle(fitness, dim, max_x, min_x) for i in range(n)]

    fitness_vals = np.array([p.fitness for p in swarm])

    if minimize:
        g_best_idx = np.argmin(fitness_vals)
    else:
        g_best_idx = np.argmax(fitness_vals)
    
    g_best_pos = swarm[g_best_idx].position.copy()
    g_best_fit = fitness_vals[g_best_idx]
    g_best_iter = 0

    res = [g_best_fit]

    for it in range(max_iter):
        for i, p in enumerate(swarm):
            r1 = np.random.rand(dim)
            r2 = np.random.rand(dim)

            # Cập nhật vận tốc cá thể
            p.velocity = w * p.velocity + c1 * r1 * (p.best_pos - p.position) + c2 * r2 * (g_best_pos - p.position)
            p.velocity = p.limit_v(p.velocity)

            # Cập nhật vị trí cá thể
            p.position = p.position + p.velocity
            p.position = p.limit_x(p.position)

            # Tính fitness cho vị trí mới
            p.fitness = fitness(p.position)

            # Cập nhật best cho cá thể
            if minimize:
                if p.fitness < p.best_fitness:
                    p.best_fitness = p.fitness
                    p.best_pos = p.position.copy()
            else:
                if p.fitness > p.best_fitness:
                    p.best_fitness = p.fitness
                    p.best_pos = p.position.copy()
            
        p_best = np.array([p.best_fitness for p in swarm])
        if minimize:
            best_idx = np.argmin(p_best)
            if p_best[best_idx] < g_best_fit:
                g_best_fit = p_best[best_idx]
                g_best_pos = swarm[best_idx].best_pos.copy()
                g_best_iter = it
        else:
            best_idx = np.argmin(p_best)
            if p_best[best_idx] > g_best_fit:
                g_best_fit = p_best[best_idx]
                g_best_pos = swarm[best_idx].best_pos.copy()
                g_best_iter = it

        res.append(g_best_fit)
    
    return {
        "best_position": g_best_pos,
        "best_fitness": g_best_fit,
        "best_iter": g_best_iter,
        "fit_res": res,
        "positions": np.array([p.position for p in swarm])
    }











