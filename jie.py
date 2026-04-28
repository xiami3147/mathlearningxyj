import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, mean_squared_error
import math
import random

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class AIVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("可视化学习平台")
        self.root.geometry("1200x750")

        self.mode = tk.StringVar(value="机器学习")

        # 机器学习专用变量
        self.ml_task = tk.StringVar(value="分类")
        self.ml_algorithm = tk.StringVar(value="逻辑回归")
        self.X = None
        self.y = None
        self.model = None
        self.ml_param_widgets = {}

        # 函数图像变量
        self.func_type = tk.StringVar(value="正比例")
        self.func_k = tk.DoubleVar(value=1.0)

        # 几何体变量
        self.geo_type = tk.StringVar(value="圆")
        self.radius = tk.DoubleVar(value=1.0)
        self.height = tk.DoubleVar(value=1.0)

        # 方块搭建变量
        self.block_count = tk.IntVar(value=10)
        self.shape_voxels = None   # 存储体素坐标列表 [(x,y,z), ...] 全部 >=0

        # 创建右侧画板容器
        self.right_frame = ttk.Frame(self.root)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.current_canvas = None

        # 创建左侧控制面板
        self.create_control_panel()

        # 默认显示机器学习界面
        self.switch_to_ml()

    def create_control_panel(self):
        control_frame = ttk.LabelFrame(self.root, text="控制面板", width=280, padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        ttk.Label(control_frame, text="功能模式:").grid(row=0, column=0, sticky=tk.W, pady=5)
        mode_combo = ttk.Combobox(control_frame, textvariable=self.mode,
                                  values=["机器学习", "函数图像", "几何体", "方块搭建","分数可视化"], state="readonly")
        mode_combo.grid(row=0, column=1, pady=5)
        mode_combo.bind("<<ComboboxSelected>>", self.on_mode_changed)

        self.dynamic_frame = ttk.Frame(control_frame)
        self.dynamic_frame.grid(row=1, column=0, columnspan=2, sticky=tk.W+tk.E, pady=10)

    def on_mode_changed(self, event=None):
        mode = self.mode.get()
        if mode == "机器学习":
            self.switch_to_ml()
        elif mode == "函数图像":
            self.switch_to_func()
        elif mode == "几何体":
            self.switch_to_geo()
        elif mode =="分数可视化":
            self.switch_to_fraction()
        else:
            self.switch_to_blocks()

    def clear_dynamic_frame(self):
        for widget in self.dynamic_frame.winfo_children():
            widget.destroy()

    def clear_right_canvas(self):
        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()
            self.current_canvas = None

    # ==================== 机器学习模块 ====================
    def switch_to_ml(self):
        self.clear_dynamic_frame()
        self.clear_right_canvas()
        fig = plt.Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=self.right_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.current_canvas = canvas
        self.ml_fig = fig
        self.ml_ax = ax

        ttk.Label(self.dynamic_frame, text="任务类型:").grid(row=0, column=0, sticky=tk.W, pady=5)
        task_combo = ttk.Combobox(self.dynamic_frame, textvariable=self.ml_task,
                                  values=["分类", "回归", "聚类"], state="readonly")
        task_combo.grid(row=0, column=1, pady=5)
        task_combo.bind("<<ComboboxSelected>>", self.on_ml_task_changed)

        ttk.Label(self.dynamic_frame, text="算法:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.ml_algo_combo = ttk.Combobox(self.dynamic_frame, textvariable=self.ml_algorithm, state="readonly")
        self.ml_algo_combo.grid(row=1, column=1, pady=5)
        self.ml_algo_combo.bind("<<ComboboxSelected>>", self.on_ml_algo_changed)

        self.ml_param_frame = ttk.LabelFrame(self.dynamic_frame, text="算法参数", padding=5)
        self.ml_param_frame.grid(row=2, column=0, columnspan=2, sticky=tk.W+tk.E, pady=10)

        ttk.Button(self.dynamic_frame, text="重新生成数据", command=self.ml_gen_data).grid(row=3, column=0, columnspan=2, pady=10)
        ttk.Button(self.dynamic_frame, text="重新训练并绘图", command=self.ml_train_plot).grid(row=4, column=0, columnspan=2, pady=5)

        self.ml_update_algo_list()
        self.ml_gen_data()
        self.ml_train_plot()

    def ml_update_algo_list(self):
        task = self.ml_task.get()
        if task == "分类":
            algos = ["逻辑回归", "K近邻"]
        elif task == "回归":
            algos = ["线性回归", "多项式回归"]
        else:
            algos = ["K-Means", "DBSCAN"]
        self.ml_algo_combo['values'] = algos
        self.ml_algorithm.set(algos[0])
        self.ml_build_param_controls()

    def on_ml_task_changed(self, event=None):
        self.ml_update_algo_list()
        self.ml_gen_data()
        self.ml_train_plot()

    def on_ml_algo_changed(self, event=None):
        self.ml_build_param_controls()
        self.ml_train_plot()

    def ml_build_param_controls(self):
        for widget in self.ml_param_frame.winfo_children():
            widget.destroy()
        self.ml_param_widgets.clear()
        task = self.ml_task.get()
        algo = self.ml_algorithm.get()
        row = 0
        if task == "分类":
            if algo == "逻辑回归":
                tk.Label(self.ml_param_frame, text="正则化强度 C:").grid(row=row, column=0, sticky=tk.W)
                var = tk.DoubleVar(value=1.0)
                scale = tk.Scale(self.ml_param_frame, from_=0.01, to=10.0, resolution=0.1, orient=tk.HORIZONTAL,
                                 variable=var, command=lambda x: self.ml_train_plot())
                scale.grid(row=row, column=1, sticky=tk.W+tk.E)
                self.ml_param_widgets['C'] = var
            elif algo == "K近邻":
                tk.Label(self.ml_param_frame, text="邻居数 K:").grid(row=row, column=0, sticky=tk.W)
                var = tk.IntVar(value=5)
                scale = tk.Scale(self.ml_param_frame, from_=1, to=20, resolution=1, orient=tk.HORIZONTAL,
                                 variable=var, command=lambda x: self.ml_train_plot())
                scale.grid(row=row, column=1, sticky=tk.W+tk.E)
                self.ml_param_widgets['n_neighbors'] = var
        elif task == "回归":
            if algo == "线性回归":
                tk.Label(self.ml_param_frame, text="无参数").grid(row=row, column=0, columnspan=2)
            elif algo == "多项式回归":
                tk.Label(self.ml_param_frame, text="多项式次数:").grid(row=row, column=0, sticky=tk.W)
                var = tk.IntVar(value=2)
                scale = tk.Scale(self.ml_param_frame, from_=1, to=5, resolution=1, orient=tk.HORIZONTAL,
                                 variable=var, command=lambda x: self.ml_train_plot())
                scale.grid(row=row, column=1, sticky=tk.W+tk.E)
                self.ml_param_widgets['degree'] = var
                row += 1
                tk.Label(self.ml_param_frame, text="正则化 alpha:").grid(row=row, column=0, sticky=tk.W)
                var2 = tk.DoubleVar(value=0.0)
                scale2 = tk.Scale(self.ml_param_frame, from_=0.0, to=2.0, resolution=0.1, orient=tk.HORIZONTAL,
                                  variable=var2, command=lambda x: self.ml_train_plot())
                scale2.grid(row=row, column=1, sticky=tk.W+tk.E)
                self.ml_param_widgets['alpha'] = var2
        elif task == "聚类":
            if algo == "K-Means":
                tk.Label(self.ml_param_frame, text="簇数量 K:").grid(row=row, column=0, sticky=tk.W)
                var = tk.IntVar(value=3)
                scale = tk.Scale(self.ml_param_frame, from_=2, to=8, resolution=1, orient=tk.HORIZONTAL,
                                 variable=var, command=lambda x: self.ml_train_plot())
                scale.grid(row=row, column=1, sticky=tk.W+tk.E)
                self.ml_param_widgets['n_clusters'] = var
            elif algo == "DBSCAN":
                tk.Label(self.ml_param_frame, text="邻域半径 eps:").grid(row=row, column=0, sticky=tk.W)
                var1 = tk.DoubleVar(value=0.5)
                scale1 = tk.Scale(self.ml_param_frame, from_=0.1, to=2.0, resolution=0.05, orient=tk.HORIZONTAL,
                                  variable=var1, command=lambda x: self.ml_train_plot())
                scale1.grid(row=row, column=1, sticky=tk.W+tk.E)
                self.ml_param_widgets['eps'] = var1
                row += 1
                tk.Label(self.ml_param_frame, text="最小样本数 min_samples:").grid(row=row, column=0, sticky=tk.W)
                var2 = tk.IntVar(value=5)
                scale2 = tk.Scale(self.ml_param_frame, from_=2, to=20, resolution=1, orient=tk.HORIZONTAL,
                                  variable=var2, command=lambda x: self.ml_train_plot())
                scale2.grid(row=row, column=1, sticky=tk.W+tk.E)
                self.ml_param_widgets['min_samples'] = var2

    def ml_gen_data(self):
        task = self.ml_task.get()
        np.random.seed()
        if task == "分类":
            self.X, self.y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                                                 n_informative=2, n_clusters_per_class=1,
                                                 flip_y=0.05, random_state=None)
        elif task == "回归":
            self.X, self.y = make_regression(n_samples=150, n_features=1, noise=10, random_state=None)
            self.y = self.y.ravel()
        else:
            self.X, _ = make_blobs(n_samples=200, centers=4, n_features=2, cluster_std=0.8, random_state=None)
            self.y = None
        self.ml_train_plot()

    def ml_train_plot(self):
        if self.mode.get() != "机器学习":
            return
        task = self.ml_task.get()
        algo = self.ml_algorithm.get()
        params = {k: v.get() for k, v in self.ml_param_widgets.items()}
        if self.X is None:
            self.ml_gen_data()
        if task != "聚类" and self.y is None:
            self.ml_gen_data()
        if task == "回归" and self.X is not None and self.X.shape[1] != 1:
            self.ml_gen_data()
        try:
            if task == "分类":
                if algo == "逻辑回归":
                    C = params.get('C', 1.0)
                    self.model = LogisticRegression(C=C, random_state=42)
                else:
                    k = params.get('n_neighbors', 5)
                    self.model = KNeighborsClassifier(n_neighbors=k)
                self.model.fit(self.X, self.y)
                self.ml_plot_classification()
            elif task == "回归":
                if algo == "线性回归":
                    self.model = LinearRegression()
                    self.model.fit(self.X, self.y)
                else:
                    degree = params.get('degree', 2)
                    alpha = params.get('alpha', 0.0)
                    if alpha > 0:
                        model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=alpha))
                    else:
                        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
                    model.fit(self.X, self.y)
                    self.model = model
                self.ml_plot_regression()
            else:
                if algo == "K-Means":
                    n_clusters = params.get('n_clusters', 3)
                    self.model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    y_pred = self.model.fit_predict(self.X)
                else:
                    eps = params.get('eps', 0.5)
                    min_samples = params.get('min_samples', 5)
                    self.model = DBSCAN(eps=eps, min_samples=min_samples)
                    y_pred = self.model.fit_predict(self.X)
                self.ml_plot_clustering(y_pred)
            self.current_canvas.draw_idle()
        except Exception as e:
            messagebox.showerror("错误", f"训练或绘图时发生错误:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def ml_plot_classification(self):
        ax = self.ml_ax
        ax.clear()
        x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        y_min, y_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
        ax.scatter(self.X[:, 0], self.X[:, 1], c=self.y, edgecolors='k', cmap=plt.cm.RdYlBu, s=40)
        ax.set_title(f"{self.ml_task.get()} - {self.ml_algorithm.get()}")
        ax.set_xlabel("特征 1")
        ax.set_ylabel("特征 2")
        y_pred = self.model.predict(self.X)
        acc = accuracy_score(self.y, y_pred)
        ax.text(0.05, 0.95, f"准确率: {acc:.2f}", transform=ax.transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        self.current_canvas.draw()

    def ml_plot_regression(self):
        ax = self.ml_ax
        ax.clear()
        X_plot = np.linspace(self.X.min() - 0.5, self.X.max() + 0.5, 300).reshape(-1, 1)
        y_plot = self.model.predict(X_plot)
        ax.scatter(self.X, self.y, alpha=0.7, label="真实数据")
        ax.plot(X_plot, y_plot, color='red', linewidth=2, label="拟合曲线")
        ax.set_title(f"{self.ml_task.get()} - {self.ml_algorithm.get()}")
        ax.set_xlabel("特征 X")
        ax.set_ylabel("目标值 y")
        ax.legend()
        y_pred = self.model.predict(self.X)
        mse = mean_squared_error(self.y, y_pred)
        ax.text(0.05, 0.95, f"MSE: {mse:.2f}", transform=ax.transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        self.current_canvas.draw()

    def ml_plot_clustering(self, y_pred):
        ax = self.ml_ax
        ax.clear()
        unique_labels = np.unique(y_pred)
        n_clusters = len([l for l in unique_labels if l != -1])
        n_noise = list(y_pred).count(-1)
        for label in unique_labels:
            if label == -1:
                color = 'gray'
                marker = 'x'
                label_txt = '噪声'
            else:
                color = None
                marker = 'o'
                label_txt = f'簇 {label}'
            mask = (y_pred == label)
            ax.scatter(self.X[mask, 0], self.X[mask, 1], c=color, marker=marker,
                       edgecolors='k', s=40, label=label_txt)
        ax.set_title(f"{self.ml_task.get()} - {self.ml_algorithm.get()}")
        ax.set_xlabel("特征 1")
        ax.set_ylabel("特征 2")
        info = f"簇数量: {n_clusters}"
        if n_noise > 0:
            info += f", 噪声点: {n_noise}"
        ax.text(0.05, 0.95, info, transform=ax.transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        ax.legend(loc='best')
        self.current_canvas.draw()

    # ==================== 函数图像模块 ====================
    def switch_to_func(self):
        self.clear_dynamic_frame()
        self.clear_right_canvas()
        fig = plt.Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(fig, master=self.right_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.current_canvas = canvas
        self.func_fig = fig
        self.func_ax = ax

        ttk.Label(self.dynamic_frame, text="函数类型:").grid(row=0, column=0, sticky=tk.W, pady=5)
        func_combo = ttk.Combobox(self.dynamic_frame, textvariable=self.func_type,
                                  values=["正比例", "反比例"], state="readonly")
        func_combo.grid(row=0, column=1, pady=5)
        func_combo.bind("<<ComboboxSelected>>", lambda e: self.plot_function())

        ttk.Label(self.dynamic_frame, text="参数 k:").grid(row=1, column=0, sticky=tk.W, pady=5)
        k_entry = ttk.Entry(self.dynamic_frame, textvariable=self.func_k, width=10)
        k_entry.grid(row=1, column=1, sticky=tk.W, pady=5)
        k_entry.bind("<KeyRelease>", lambda e: self.plot_function())

        info_text = "正比例: y = kx\n反比例: y = k/x (x≠0)"
        ttk.Label(self.dynamic_frame, text=info_text, justify=tk.LEFT).grid(row=2, column=0, columnspan=2, pady=10)
        ttk.Button(self.dynamic_frame, text="刷新图像", command=self.plot_function).grid(row=3, column=0, columnspan=2, pady=5)
        self.plot_function()

    def plot_function(self):
        if self.mode.get() != "函数图像":
            return
        ax = self.func_ax
        ax.clear()
        func = self.func_type.get()
        k = self.func_k.get()
        x = np.linspace(-5, 5, 400)
        if func == "正比例":
            y = k * x
            ax.plot(x, y, label=f'y = {k:.2f}x', color='blue')
            ax.set_ylim(-10, 10)
        else:
            x1 = np.linspace(-5, -0.1, 200)
            x2 = np.linspace(0.1, 5, 200)
            y1 = k / x1
            y2 = k / x2
            ax.plot(x1, y1, 'b', label=f'y = {k:.2f}/x')
            ax.plot(x2, y2, 'b')
            ax.set_ylim(-20, 20)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title(f"{func}函数图像  (k={k:.2f})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        self.current_canvas.draw()

    # ==================== 几何体模块 ====================
    def switch_to_geo(self):
        self.clear_dynamic_frame()
        self.clear_right_canvas()
        fig = plt.Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        canvas = FigureCanvasTkAgg(fig, master=self.right_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.current_canvas = canvas
        self.geo_fig = fig
        self.geo_ax = ax

        ttk.Label(self.dynamic_frame, text="几何体类型:").grid(row=0, column=0, sticky=tk.W, pady=5)
        geo_combo = ttk.Combobox(self.dynamic_frame, textvariable=self.geo_type,
                                 values=["圆", "圆柱", "圆锥"], state="readonly")
        geo_combo.grid(row=0, column=1, pady=5)
        geo_combo.bind("<<ComboboxSelected>>", lambda e: self.update_geo_visibility())

        ttk.Label(self.dynamic_frame, text="半径 r:").grid(row=1, column=0, sticky=tk.W, pady=5)
        r_entry = ttk.Entry(self.dynamic_frame, textvariable=self.radius, width=10)
        r_entry.grid(row=1, column=1, sticky=tk.W, pady=5)
        r_entry.bind("<KeyRelease>", lambda e: self.plot_geometry())

        self.geo_height_label = ttk.Label(self.dynamic_frame, text="高度 h:")
        self.geo_height_entry = ttk.Entry(self.dynamic_frame, textvariable=self.height, width=10)

        self.geo_result_label = ttk.Label(self.dynamic_frame, text="", justify=tk.LEFT)
        self.geo_result_label.grid(row=4, column=0, columnspan=2, pady=10)

        ttk.Label(self.dynamic_frame, text="💡 鼠标拖拽旋转视图", foreground="blue").grid(row=5, column=0, columnspan=2, pady=5)

        self.update_geo_visibility()
        self.plot_geometry()

    def update_geo_visibility(self):
        geo = self.geo_type.get()
        if geo == "圆":
            self.geo_height_label.grid_forget()
            self.geo_height_entry.grid_forget()
        else:
            self.geo_height_label.grid(row=2, column=0, sticky=tk.W, pady=5)
            self.geo_height_entry.grid(row=2, column=1, sticky=tk.W, pady=5)
        self.plot_geometry()

    def plot_geometry(self):
        if self.mode.get() != "几何体":
            return
        geo = self.geo_type.get()
        r = self.radius.get()
        h = self.height.get()
        if r <= 0:
            self.geo_result_label.config(text="半径必须为正数")
            return
        ax = self.geo_ax
        ax.clear()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        if geo == "圆":
            theta = np.linspace(0, 2*np.pi, 100)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = np.zeros_like(theta)
            ax.plot(x, y, z, color='blue', linewidth=2, label=f'圆 (r={r:.2f})')
            ax.set_title(f"圆 (半径 r={r:.2f})")
            lim = r + 0.5
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-lim, lim)
            area = math.pi * r * r
            self.geo_result_label.config(text=f"圆的面积: π × r² = {math.pi:.3f} × {r:.2f}² = {area:.3f}")

        elif geo == "圆柱":
            theta = np.linspace(0, 2*np.pi, 50)
            z_plane = np.linspace(0, h, 50)
            theta_grid, z_grid = np.meshgrid(theta, z_plane)
            x_grid = r * np.cos(theta_grid)
            y_grid = r * np.sin(theta_grid)
            ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5, color='cyan', edgecolor='none')
            r_plane = np.linspace(0, r, 30)
            theta_plane = np.linspace(0, 2*np.pi, 50)
            r_grid, theta_grid2 = np.meshgrid(r_plane, theta_plane)
            x_top = r_grid * np.cos(theta_grid2)
            y_top = r_grid * np.sin(theta_grid2)
            z_top = np.full_like(x_top, h)
            ax.plot_surface(x_top, y_top, z_top, alpha=0.6, color='orange')
            z_bottom = np.zeros_like(x_top)
            ax.plot_surface(x_top, y_top, z_bottom, alpha=0.6, color='orange')
            ax.set_title(f"圆柱 (r={r:.2f}, h={h:.2f})")
            lim = max(r, h/2) + 0.5
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-0.5, h+0.5)
            area_base = math.pi * r * r
            volume = area_base * h
            surface_area = 2 * area_base + 2 * math.pi * r * h
            self.geo_result_label.config(text=f"圆柱:\n底面积 = πr² = {area_base:.3f}\n体积 = {volume:.3f}\n表面积 = {surface_area:.3f}")

        else:  # 圆锥
            theta = np.linspace(0, 2*np.pi, 50)
            z = np.linspace(0, h, 50)
            z_grid, theta_grid = np.meshgrid(z, theta)
            r_cone = r * (1 - z_grid / h)
            x_cone = r_cone * np.cos(theta_grid)
            y_cone = r_cone * np.sin(theta_grid)
            ax.plot_surface(x_cone, y_cone, z_grid, alpha=0.5, color='green', edgecolor='none')
            r_disk = np.linspace(0, r, 30)
            theta_disk = np.linspace(0, 2*np.pi, 50)
            r_grid, theta_grid2 = np.meshgrid(r_disk, theta_disk)
            x_bottom = r_grid * np.cos(theta_grid2)
            y_bottom = r_grid * np.sin(theta_grid2)
            z_bottom = np.zeros_like(x_bottom)
            ax.plot_surface(x_bottom, y_bottom, z_bottom, alpha=0.6, color='lightblue')
            ax.set_title(f"圆锥 (r={r:.2f}, h={h:.2f})")
            lim = max(r, h/2) + 0.5
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-0.5, h+0.5)
            area_base = math.pi * r * r
            volume = area_base * h / 3
            slant_height = math.sqrt(r*r + h*h)
            surface_area = area_base + math.pi * r * slant_height
            self.geo_result_label.config(text=f"圆锥:\n底面积 = πr² = {area_base:.3f}\n体积 = {volume:.3f}\n表面积 = πr² + πrl = {surface_area:.3f}")

        ax.view_init(elev=25, azim=-60)
        self.current_canvas.draw()

    # ==================== 方块搭建模块（修正版：第一象限+规整刻度） ====================
    def switch_to_blocks(self):
        self.clear_dynamic_frame()
        self.clear_right_canvas()

        # 创建 Figure，布局：上方3D子图，下方三个视图水平排列
        fig = plt.Figure(figsize=(8, 8), dpi=100, constrained_layout=True)
        gs = fig.add_gridspec(2, 3, height_ratios=[2, 1], hspace=0.3, wspace=0.3)
        ax3d = fig.add_subplot(gs[0, :], projection='3d')
        ax_front = fig.add_subplot(gs[1, 0])   # 正视图
        ax_side  = fig.add_subplot(gs[1, 1])   # 左视图
        ax_top   = fig.add_subplot(gs[1, 2])   # 俯视图

        canvas = FigureCanvasTkAgg(fig, master=self.right_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.current_canvas = canvas
        self.block_fig = fig
        self.block_ax3d = ax3d
        self.block_ax_front = ax_front
        self.block_ax_side = ax_side
        self.block_ax_top = ax_top

        # 控件
        ttk.Label(self.dynamic_frame, text="方块数量:").grid(row=0, column=0, sticky=tk.W, pady=5)
        count_spinbox = ttk.Spinbox(self.dynamic_frame, from_=1, to=100, textvariable=self.block_count, width=10)
        count_spinbox.grid(row=0, column=1, sticky=tk.W, pady=5)
        count_spinbox.bind("<KeyRelease>", lambda e: self.generate_random_shape())

        ttk.Button(self.dynamic_frame, text="随机生成形状", command=self.generate_random_shape).grid(row=1, column=0, columnspan=2, pady=10)

        info_text = "输入方块数量。"
        ttk.Label(self.dynamic_frame, text=info_text, justify=tk.LEFT, wraplength=250).grid(row=2, column=0, columnspan=2, pady=10)

        self.generate_random_shape()

    def generate_random_shape(self):
        n = self.block_count.get()
        if n < 1:
            messagebox.showwarning("警告", "方块数量必须至少为1")
            return
        # 随机游走生成连通形状，起始点(0,0,0) 确保非负
        voxels = set()
        voxels.add((0, 0, 0))
        directions = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
        while len(voxels) < n:
            neighbors = set()
            for (x,y,z) in voxels:
                for dx,dy,dz in directions:
                    nx, ny, nz = x+dx, y+dy, z+dz
                    # 允许任何坐标，但后续会平移至非负，这里先不限制
                    if (nx, ny, nz) not in voxels:
                        neighbors.add((nx, ny, nz))
            if not neighbors:
                break
            new_voxel = random.choice(list(neighbors))
            voxels.add(new_voxel)
        # 将全部坐标平移到第一象限（最小坐标变为0）
        xs = [v[0] for v in voxels]
        ys = [v[1] for v in voxels]
        zs = [v[2] for v in voxels]
        min_x, min_y, min_z = min(xs), min(ys), min(zs)
        translated = [(x - min_x, y - min_y, z - min_z) for (x,y,z) in voxels]
        self.shape_voxels = translated
        self.plot_three_views()

    def plot_three_views(self):
        if self.mode.get() != "方块搭建" or not self.shape_voxels:
            return
        voxels = self.shape_voxels

        xs = [v[0] for v in voxels]
        ys = [v[1] for v in voxels]
        zs = [v[2] for v in voxels]
        max_x, max_y, max_z = max(xs), max(ys), max(zs)
        size_x, size_y, size_z = max_x+1, max_y+1, max_z+1

        # 构建三视图矩阵（行=垂直方向，列=水平方向）
        # 正视图：垂直 Z，水平 X
        front = np.zeros((size_z, size_x), dtype=bool)
        # 左视图：垂直 Z，水平 Y
        left  = np.zeros((size_z, size_y), dtype=bool)
        # 俯视图：垂直 Y，水平 X
        top   = np.zeros((size_y, size_x), dtype=bool)

        for (x,y,z) in voxels:
            front[z, x] = True
            left[z,y] = True
            top[y, x] = True

        # 清空并绘制3D图（保持不变）
        self.block_ax3d.clear()
        def add_cube(ax, x, y, z):
            vertices = [
                [[x,y,z], [x+1,y,z], [x+1,y+1,z], [x,y+1,z]],
                [[x,y,z+1], [x+1,y,z+1], [x+1,y+1,z+1], [x,y+1,z+1]],
                [[x,y,z], [x+1,y,z], [x+1,y,z+1], [x,y,z+1]],
                [[x+1,y,z], [x+1,y+1,z], [x+1,y+1,z+1], [x+1,y,z+1]],
                [[x,y,z], [x,y+1,z], [x,y+1,z+1], [x,y,z+1]],
                [[x,y+1,z], [x+1,y+1,z], [x+1,y+1,z+1], [x,y+1,z+1]]
            ]
            collection = Poly3DCollection(vertices, facecolors='royalblue', edgecolors='black', linewidths=0.5, alpha=1.0)
            ax.add_collection3d(collection)
        for (x,y,z) in voxels:
            add_cube(self.block_ax3d, x, y, z)
        self.block_ax3d.set_xlabel('X')
        self.block_ax3d.set_ylabel('Y')
        self.block_ax3d.set_zlabel('Z')
        self.block_ax3d.set_title(f"3D形状 (共{len(voxels)}个方块)")
        self.block_ax3d.set_xlim(-0.5, max_x+1.5)
        self.block_ax3d.set_ylim(-0.5, max_y+1.5)
        self.block_ax3d.set_zlim(-0.5, max_z+1.5)
        self.block_ax3d.set_xticks(np.arange(0, max_x+2, 1))
        self.block_ax3d.set_yticks(np.arange(0, max_y+2, 1))
        self.block_ax3d.set_zticks(np.arange(0, max_z+2, 1))

        # 绘制三视图（每个视图强制正方形格子）
        self.block_ax_front.clear()
        self._draw_grid(self.block_ax_front, front, "正视图", "X", "Z")
        self.block_ax_side.clear()
        self._draw_grid(self.block_ax_side, left, "右视图", "Y", "Z")
        self.block_ax_top.clear()
        self._draw_grid(self.block_ax_top, top, "俯视图", "X", "Y")

        self.current_canvas.draw()

    def _draw_grid(self, ax, grid, title, xlabel, ylabel):
        """
        绘制黑白网格，每个格子强制为正方形。
        grid 形状 (rows, cols)  rows=垂直方向格数，cols=水平方向格数。
        """
        data = np.where(grid, 1, 0)   # 1黑 0白
        ax.imshow(data, origin='lower', cmap='gray', interpolation='none', aspect='equal')
        rows, cols = grid.shape
        # 设置刻度位置和标签
        ax.set_xticks(np.arange(cols))
        ax.set_yticks(np.arange(rows))
        ax.set_xticklabels(np.arange(cols))
        ax.set_yticklabels(np.arange(rows))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        # 添加网格线
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        ax.tick_params(which='minor', size=0)
        # 关键：强制每个轴的刻度单位长度相等，使格子显示为正方形
        ax.set_aspect('equal')
        # ==================== 分数可视化模块 ====================
    def switch_to_fraction(self):
        self.clear_dynamic_frame()
        self.clear_right_canvas()
        
        fig = plt.Figure(figsize=(8, 6), dpi=100)
        # 左右两个子图：圆形和矩形
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        canvas = FigureCanvasTkAgg(fig, master=self.right_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.current_canvas = canvas
        self.fraction_fig = fig
        self.fraction_ax_circle = ax1
        self.fraction_ax_rect = ax2
        
        # 分数变量
        self.fraction_numerator = tk.IntVar(value=1)
        self.fraction_denominator = tk.IntVar(value=4)
        
        ttk.Label(self.dynamic_frame, text="分子:").grid(row=0, column=0, sticky=tk.W, pady=5)
        num_spin = ttk.Spinbox(self.dynamic_frame, from_=0, to=100, textvariable=self.fraction_numerator, width=6)
        num_spin.grid(row=0, column=1, sticky=tk.W, pady=5)
        num_spin.bind("<KeyRelease>", lambda e: self.plot_fraction())
        
        ttk.Label(self.dynamic_frame, text="分母:").grid(row=1, column=0, sticky=tk.W, pady=5)
        den_spin = ttk.Spinbox(self.dynamic_frame, from_=1, to=100, textvariable=self.fraction_denominator, width=6)
        den_spin.grid(row=1, column=1, sticky=tk.W, pady=5)
        den_spin.bind("<KeyRelease>", lambda e: self.plot_fraction())
        
        self.fraction_info_label = ttk.Label(self.dynamic_frame, text="", justify=tk.LEFT)
        self.fraction_info_label.grid(row=2, column=0, columnspan=2, pady=10)
        
        ttk.Button(self.dynamic_frame, text="化简分数", command=self.simplify_fraction).grid(row=3, column=0, columnspan=2, pady=5)
        
        self.plot_fraction()
    
    def simplify_fraction(self):
        from math import gcd
        num = self.fraction_numerator.get()
        den = self.fraction_denominator.get()
        if den == 0:
            return
        g = gcd(num, den)
        self.fraction_numerator.set(num // g)
        self.fraction_denominator.set(den // g)
        self.plot_fraction()
    
    def plot_fraction(self):
        if self.mode.get() != "分数可视化":
            return
        num = self.fraction_numerator.get()
        den = self.fraction_denominator.get()
        if den <= 0:
            den = 1
            self.fraction_denominator.set(1)
        if num < 0:
            num = 0
            self.fraction_numerator.set(0)
        
        # 显示小数和百分比
        value = num / den
        percent = value * 100
        info = f"数值: {value:.4f}\n百分比: {percent:.2f}%"
        from math import gcd
        g = gcd(num, den)
        if g > 1:
            info += f"\n最简分数: {num//g}/{den//g}"
        self.fraction_info_label.config(text=info)
        
        # 圆形分数盘（饼图）
        ax_c = self.fraction_ax_circle
        ax_c.clear()
        if num == 0:
            ax_c.pie([1], colors=['white'], startangle=90)
            ax_c.set_title("0")
        elif num >= den:
            ax_c.pie([1], colors=['cornflowerblue'], startangle=90)
            ax_c.set_title("1")
        else:
            colors = ['cornflowerblue', 'lightgray']
            ax_c.pie([num, den-num], colors=colors, startangle=90)
            ax_c.text(0, 0, f"{num}/{den}", ha='center', va='center', fontsize=16)
        ax_c.set_title(f"圆形分数盘  {num}/{den}")
        ax_c.axis('equal')
        
        # 矩形分数条
        ax_r = self.fraction_ax_rect
        ax_r.clear()
        if num == 0:
            ax_r.add_patch(plt.Rectangle((0,0), 1, 1, color='white', edgecolor='black', linewidth=2))
        elif num >= den:
            ax_r.add_patch(plt.Rectangle((0,0), 1, 1, color='cornflowerblue', edgecolor='black', linewidth=2))
        else:
            frac = num / den
            ax_r.add_patch(plt.Rectangle((0,0), frac, 1, color='cornflowerblue', edgecolor='black', linewidth=2))
            ax_r.add_patch(plt.Rectangle((frac,0), 1-frac, 1, color='white', edgecolor='black', linewidth=2))
        ax_r.set_xlim(0, 1)
        ax_r.set_ylim(0, 1)
        ax_r.set_aspect('equal')
        ax_r.set_xticks([])
        ax_r.set_yticks([])
        ax_r.set_title(f"矩形分数条  {num}/{den}")
        
        self.current_canvas.draw()
  
    
    

if __name__ == "__main__":
    root = tk.Tk()
    app = AIVisualizer(root)
    root.mainloop()
