# 使用更小的基础镜像，避免网络问题
FROM openjdk:17-alpine

# 安装必要的包（Alpine 使用 apk）
RUN apk add --no-cache python3 py3-pip bash

# 创建软链接
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip || true

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt ./

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露端口
EXPOSE 8888

# 挂载卷
VOLUME /app

# 环境变量
ENV JUPYTER_ENABLE_LAB=yes

# 启动命令
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app"]