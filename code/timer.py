# 李炳锋24级
import tkinter as tk
from tkinter import ttk
from datetime import datetime, timedelta
import json
import os
from tkcalendar import DateEntry
import babel
import babel.numbers
import requests
from tkinter import messagebox

REFRESH_TIME = 5000
host_ip = "10.129.240.109"
url = f"http://{host_ip}:5000/"

class LoginWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent.root)
        self.parent = parent
        self.title("用户登录")
        self.geometry("300x200")
        
        tk.Label(self, text="用户名:").pack(pady=5)
        self.username = tk.Entry(self)
        self.username.pack()
        
        tk.Label(self, text="密码:").pack(pady=5)
        self.password = tk.Entry(self, show="*")
        self.password.pack()
        
        tk.Button(self, text="登录", command=self.login).pack(pady=10)
        tk.Button(self, text="注册", command=self.register).pack()

    def login(self):
        username = self.username.get()
        password = self.password.get()
        
        response = requests.post(
            f"{url}login",
            json={"username": username, "password": password},
            verify=False  # 正式环境应使用有效证书
        )
        
        if response.status_code == 200:
            self.parent.auth = (username, password)
            messagebox.showerror("好耶！", "登陆成功")
            self.destroy()
        else:
            messagebox.showerror("错误", "登录失败")

    def register(self):
        username = self.username.get()
        password = self.password.get()
        
        response = requests.post(
            f"{url}register",
            json={"username": username, "password": password},
            verify=False
        )
        
        if response.status_code == 201:
            messagebox.showinfo("成功", "注册成功")
        else:
            messagebox.showerror("错误", response.json().get("error", "注册失败"))

class SettingsWindow(tk.Toplevel):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.title("软件设置")
        self.geometry("300x200")
        
        # 刷新间隔设置
        tk.Label(self, text="刷新间隔(毫秒):").pack(pady=5)
        self.refresh_entry = tk.Entry(self)
        self.refresh_entry.insert(0, str(self.app.settings["refresh_interval"]))
        self.refresh_entry.pack()
        
        # 窗口标题设置
        tk.Label(self, text="窗口标题:").pack(pady=5)
        self.title_entry = tk.Entry(self)
        self.title_entry.insert(0, self.app.settings["window_title"])
        self.title_entry.pack()
        
        # 保存按钮
        tk.Button(self, text="保存设置", command=self.save_settings,
                 bg="#2196F3", fg="white").pack(pady=10)

    def save_settings(self):
        """保存设置到文件"""
        try:
            # 验证输入
            new_interval = int(self.refresh_entry.get())
            if new_interval < 1000:
                raise ValueError("刷新间隔不能小于1000毫秒")
            
            new_title = self.title_entry.get().strip()
            if not new_title:
                raise ValueError("标题不能为空")
            
            # 更新设置
            self.app.settings.update({
                "refresh_interval": new_interval,
                "window_title": new_title
            })
            
            # 应用新设置
            self.app.root.title(new_title)
            self.app.save_settings()
            self.app.schedule_update()  # 重新设置刷新计时器
            self.destroy()
            
        except ValueError as e:
            print(f"设置错误: {str(e)}")

class EnhancedTodoApp:
    def __init__(self, root):
        self.root = root
        self.auth = None

        self.tasks = []
        self.max_priority = 100
        self.settings_file = "settings.json"
        self.settings = self.load_settings()  # 新增设置加载
        
        self.create_widgets()
        self.show_login()

        if self.auth:
            self.root.title(f'{self.settings["window_title"]} - {self.auth[0]}')
            
        self.load_data()
        self.schedule_update()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def show_login(self):
        login_window = LoginWindow(self)
        self.root.wait_window(login_window)

    def create_widgets(self):
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 在原代码的main_frame之后添加控制面板
        control_frame = tk.Frame(self.root, padx=10, pady=5)
        control_frame.pack(fill=tk.X)
        
        # 积分显示
        self.points_label = tk.Label(control_frame, 
                                    text=f"当前积分: {self.settings['user_points']}",
                                    font=("宋体", 10))
        self.points_label.pack(side=tk.LEFT)
        
        # 设置按钮
        tk.Button(control_frame, text="设置", 
                command=lambda: SettingsWindow(self.root, self),
                bg="#9C27B0", fg="white").pack(side=tk.RIGHT)

        input_frame = tk.LabelFrame(main_frame, text="新建任务", padx=10, pady=10)
        input_frame.pack(fill=tk.X)

        tk.Label(input_frame, text="任务名称:").grid(row=0, column=0, sticky="w")
        self.task_name = tk.Entry(input_frame, width=25)
        self.task_name.grid(row=0, column=1, padx=5, sticky="ew")

        self.input_mode = tk.StringVar(value="rate")
        tk.Radiobutton(input_frame, text="设置增长率", variable=self.input_mode,
                      value="rate", command=self.toggle_input_mode).grid(row=1, column=0, sticky="w")
        tk.Radiobutton(input_frame, text="设置截止时间", variable=self.input_mode,
                      value="deadline", command=self.toggle_input_mode).grid(row=1, column=1, sticky="w")

        tk.Label(input_frame, text="初始优先级:").grid(row=2, column=0, sticky="w")
        self.initial_priority = tk.Entry(input_frame, width=25)
        self.initial_priority.grid(row=2, column=1, padx=5, sticky="ew")

        self.rate_frame = tk.Frame(input_frame)
        self.deadline_frame = tk.Frame(input_frame)
        
        tk.Label(self.rate_frame, text="每小时增加:").pack(side=tk.LEFT)
        self.rate = tk.Entry(self.rate_frame, width=10)
        self.rate.pack(side=tk.LEFT, padx=5)
        
        tk.Label(self.deadline_frame, text="日期:").pack(side=tk.LEFT)
        self.deadline_date = DateEntry(self.deadline_frame)
        self.deadline_date.pack(side=tk.LEFT, padx=5)
        tk.Label(self.deadline_frame, text="时间:").pack(side=tk.LEFT)
        self.deadline_time = tk.Entry(self.deadline_frame, width=8)
        self.deadline_time.insert(0, "23:59")
        self.deadline_time.pack(side=tk.LEFT)

        self.toggle_input_mode()
        
        tk.Button(input_frame, text="添加任务", command=self.add_task,
                 bg="#4CAF50", fg="white").grid(row=5, columnspan=2, pady=10)

        list_frame = tk.LabelFrame(main_frame, text="任务列表", padx=10, pady=10)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(list_frame)
        self.scrollbar = tk.Scrollbar(list_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas)

        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

    def toggle_input_mode(self):
        if self.input_mode.get() == "rate":
            self.rate_frame.grid(row=3, columnspan=2, pady=5, sticky="w")
            self.deadline_frame.grid_forget()
        else:
            self.deadline_frame.grid(row=4, columnspan=2, pady=5, sticky="w")
            self.rate_frame.grid_forget()

    def add_task(self):
        try:
            task = {
                "name": self.task_name.get(),
                "initial": float(self.initial_priority.get()),
                "created": datetime.now().isoformat(),
                "use_deadline": (self.input_mode.get() == "deadline")
            }

            if task["use_deadline"]:
                deadline_str = f"{self.deadline_date.get_date()} {self.deadline_time.get()}"
                deadline = datetime.strptime(deadline_str, "%Y-%m-%d %H:%M")
                if deadline <= datetime.now():
                    raise ValueError("截止时间必须晚于当前时间")
                
                time_diff = (deadline - datetime.now()).total_seconds() / 3600
                task["rate"] = (self.max_priority - task["initial"]) / time_diff
                task["deadline"] = deadline.isoformat()
            else:
                task["rate"] = float(self.rate.get())
                task["deadline"] = None

        except ValueError as e:
            print(f"输入错误: {e}")
            return

        self.tasks.append(task)
        self.task_name.delete(0, tk.END)
        self.initial_priority.delete(0, tk.END)
        self.rate.delete(0, tk.END)
        self.save_data()
        self.refresh_list()

    def get_current_priority(self, task):
        created = datetime.fromisoformat(task["created"])
        hours = (datetime.now() - created).total_seconds() / 3600
        calculated = task["initial"] + task["rate"] * hours
        return min(calculated, self.max_priority)

    def refresh_list(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        # 关键修改：保存原始索引
        sorted_tasks = sorted(
            [(self.get_current_priority(t), t, idx) for idx, t in enumerate(self.tasks)],
            key=lambda x: -x[0]
        )

        for idx, (priority, task, original_idx) in enumerate(sorted_tasks):
            ratio = priority / self.max_priority
            
            if ratio <= 0.5:
                red = int(510 * ratio)
                green = 255
            else:
                red = 255
                green = int(510 * (1 - ratio))
            blue = 0
            red = max(0, min(255, red))
            green = max(0, min(255, green))
            color = f"#{red:02x}{green:02x}{blue:02x}"

            entry_frame = tk.Frame(self.scrollable_frame, bg=color, pady=5, padx=10)
            entry_frame.pack(fill=tk.X, pady=2)

            info_frame = tk.Frame(entry_frame, bg=color)
            info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            tk.Label(info_frame, text=task["name"], bg=color,
                    font=("微软雅黑", 12)).pack(anchor="w")
            
            deadline_text = ""
            if task["use_deadline"]:
                deadline = datetime.fromisoformat(task["deadline"])
                deadline_text = f" | 截止时间: {deadline.strftime('%Y-%m-%d %H:%M')}"
                
            status_text = (f"当前优先级: {priority:.3f}/{self.max_priority}"
                          f" | 增长率: {task['rate']:.2f}/小时{deadline_text}")
            
            tk.Label(info_frame, text=status_text, bg=color,
                    font=("宋体", 9)).pack(anchor="w")

            # 关键修改：绑定原始索引
            del_btn = tk.Button(entry_frame, text="×", fg="red",
                                command=lambda oi=original_idx: self.delete_task(oi),
                                font=("Arial", 14), bd=0)
            del_btn.pack(side=tk.RIGHT)


    def delete_task(self, index):
        """删除任务并计算积分"""
        if 0 <= index < len(self.tasks):
            task = self.tasks[index]
            completed_early = False
            
            # 检查是否在DDL前完成
            if task["use_deadline"]:
                deadline = datetime.fromisoformat(task["deadline"])
                completed_early = datetime.now() < deadline
            
            # 检查优先级是否未达上限
            current_priority = self.get_current_priority(task)
            below_max = current_priority < self.max_priority
            
            if completed_early or below_max:
                self.settings["user_points"] += 1
                print(f"任务完成，+1分 (当前积分: {self.settings['user_points']})")
            else:
                self.settings["user_points"] = self.settings["user_points"] - 5
                print(f"任务超时，-5分 (当前积分: {self.settings['user_points']})")
            
            del self.tasks[index]
            self.save_data()
            self.save_settings()  # 保存积分
            self.refresh_list()

    def schedule_update(self):
        """使用设置中的刷新间隔"""
        self.refresh_list()
        self.points_label.config(text=f"当前积分: {self.settings['user_points']}")
        self.root.after(self.settings["refresh_interval"], self.schedule_update)

    def load_data(self):
        """增强版数据加载方法"""
        if not self.auth:
            print("无权限")
            return
        
        username, password = self.auth
        response = requests.get(
            f"{url}todo-data",
            auth=(username, password),
            verify=False
        )

        if response.status_code == 200:
            data = response.json()
        else:
            print(f"Error: {self.response.status_code}, Unable to fetch data from the server.")

        try:
            raw_data = data

            # 数据验证和迁移
            validated_tasks = []
            for task in raw_data:
                try:
                    # 必须字段检查
                    required_fields = ["name", "initial", "rate", "created"]
                    if not all(field in task for field in required_fields):
                        print(f"发现不完整任务: {task.get('name', '无名任务')}，已跳过")
                        continue

                    # 转换时间格式
                    created = datetime.fromisoformat(task["created"])
                    deadline = None
                    if task.get("deadline"):
                        try:
                            deadline = datetime.fromisoformat(task["deadline"])
                        except ValueError:
                            pass

                    validated_tasks.append({
                        "name": task["name"],
                        "initial": float(task["initial"]),
                        "rate": float(task["rate"]),
                        "created": created.isoformat(),
                        "use_deadline": task.get("use_deadline", False),
                        "deadline": deadline.isoformat() if deadline else None
                    })
                except Exception as e:
                    print(f"处理任务数据时出错: {str(e)}")
                    continue

            self.tasks = validated_tasks
            self.refresh_list()
            print(f"成功加载 {len(self.tasks)} 个任务")

        except json.JSONDecodeError:
            print("错误：JSON文件格式不正确，请检查文件内容")
        except Exception as e:
            print(f"加载数据时发生未知错误: {str(e)}")

    def save_data(self):
        """增强版数据保存方法"""

        if not self.auth:
            print("无权限")
            return
            
        username, password = self.auth
        try:
            data_to_save = []
            for task in self.tasks:
                # 确保所有datetime对象转为字符串
                saved_task = task.copy()
                saved_task["created"] = datetime.fromisoformat(task["created"]).isoformat()
                if task["deadline"]:
                    saved_task["deadline"] = datetime.fromisoformat(task["deadline"]).isoformat()
                data_to_save.append(saved_task)

            response = requests.post(
                f"{url}todo-data",
                auth=(username, password),
                json=data_to_save,
                verify=False
            )

            if response.status_code == 200:
                print("数据上传成功！")
            else:
                print(f"错误: {response.status_code}, 未能上传！")

        except Exception as e:
            print(f"保存数据时出错: {str(e)}")

    def load_settings(self):
        """加载设置文件"""
        default_settings = {
            "refresh_interval": 5000,  # 默认5秒
            "window_title": "计划表 Pro",
            "user_points": 0
        }
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    return {**default_settings, **loaded}
            return default_settings
        except Exception as e:
            print(f"加载设置失败: {str(e)}")
            return default_settings

    def save_settings(self):
        """保存设置文件"""
        try:
            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存设置失败: {str(e)}")

    def on_close(self):
        self.save_data()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("600x800")
    app = EnhancedTodoApp(root)
    root.mainloop()