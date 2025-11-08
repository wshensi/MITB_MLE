# last updated Oct 2025
# Using official OpenJDK 17 image to avoid apt package issues
FROM openjdk:17-slim

# Set non-interactive mode to prevent apt from hanging
ENV DEBIAN_FRONTEND=noninteractive

# Replace apt sources with domestic mirrors (for faster download)
RUN sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list

# Install Python, pip, procps (for ps command) and bash
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        procps \
        bash \
        curl \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create python / pip command symlinks
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Upgrade pip to latest version
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy dependency files to container
COPY requirements.txt ./

# Configure pip to use domestic source and install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

# Expose JupyterLab default port
EXPOSE 8888

# Mount working directory for notebooks
VOLUME /app

# Enable JupyterLab
ENV JUPYTER_ENABLE_LAB=yes

# Startup command: Run JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/app"]