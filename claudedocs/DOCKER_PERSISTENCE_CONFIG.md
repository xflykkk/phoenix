# Phoenix Docker 数据持久化配置说明

## 📋 概述

本文档说明如何在不支持 docker-compose 的环境中部署 Phoenix，并确保数据持久化。

## ✅ 已完成的 Dockerfile 优化

Dockerfile 已针对 `data:/phoenix/workdir` 挂载格式进行优化：

### 关键配置
- **工作目录**: `/phoenix/workdir`
- **环境变量**: `PHOENIX_WORKING_DIR=/phoenix/workdir` (已内置)
- **VOLUME 声明**: `VOLUME ["/phoenix/workdir"]`
- **目录权限**: 已设置为 777，支持 nonroot 用户运行

## 🚀 部署配置

### 方式 1: 使用 data:/phoenix/workdir 挂载格式

如果您的平台支持这种格式（如某些云平台或 PaaS），配置如下：

```yaml
# 平台配置示例
volumes:
  - data:/phoenix/workdir

environment:
  # 可选：PHOENIX_WORKING_DIR 已在 Dockerfile 中设置为默认值
  # - PHOENIX_WORKING_DIR=/phoenix/workdir
```

### 方式 2: 使用主机路径挂载

```bash
docker run -d \
  --name phoenix \
  -p 6006:6006 \
  -p 4317:4317 \
  -v /data/phoenix/workdir:/phoenix/workdir \
  your-phoenix-image:latest
```

### 方式 3: 使用 Docker Named Volume

```bash
# 创建 volume
docker volume create phoenix-data

# 运行容器
docker run -d \
  --name phoenix \
  -p 6006:6006 \
  -p 4317:4317 \
  -v phoenix-data:/phoenix/workdir \
  your-phoenix-image:latest
```

### 方式 4: Kubernetes 配置

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: phoenix-workdir-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: phoenix
spec:
  template:
    spec:
      containers:
      - name: phoenix
        image: your-phoenix-image:latest
        ports:
        - containerPort: 6006
        - containerPort: 4317
        volumeMounts:
        - name: workdir
          mountPath: /phoenix/workdir
      volumes:
      - name: workdir
        persistentVolumeClaim:
          claimName: phoenix-workdir-pvc
```

## 📊 存储内容说明

`/phoenix/workdir` 目录存储以下内容：

### SQLite 模式（默认）
- 📁 `phoenix.db` - 主数据库文件（traces, evals, experiments）
- 📁 导出的数据集文件
- 📁 临时文件和缓存

### PostgreSQL 模式
如果使用外部 PostgreSQL，需要额外配置：

```bash
docker run -d \
  --name phoenix \
  -p 6006:6006 \
  -p 4317:4317 \
  -e PHOENIX_SQL_DATABASE_URL=postgresql://user:pass@host:5432/dbname \
  -v /data/phoenix/workdir:/phoenix/workdir \
  your-phoenix-image:latest
```

在 PostgreSQL 模式下，`/phoenix/workdir` 主要存储：
- 📁 导出的数据集文件
- 📁 临时文件和缓存
- 📁 应用级文件

## 🔍 验证持久化配置

### 测试步骤

```bash
# 1. 启动容器
docker run -d --name phoenix-test \
  -v /data/phoenix:/phoenix/workdir \
  -p 6006:6006 \
  your-image:latest

# 2. 创建测试文件
docker exec phoenix-test sh -c "echo 'persistence-test' > /phoenix/workdir/test.txt"

# 3. 在主机验证
cat /data/phoenix/test.txt
# 应该输出: persistence-test

# 4. 删除并重建容器
docker rm -f phoenix-test
docker run -d --name phoenix-test \
  -v /data/phoenix:/phoenix/workdir \
  -p 6006:6006 \
  your-image:latest

# 5. 验证数据仍存在
docker exec phoenix-test cat /phoenix/workdir/test.txt
# 应该输出: persistence-test
```

### 验证检查点

✅ **成功的标志：**
- 容器重启后数据仍然存在
- 主机目录中可以看到 Phoenix 创建的文件
- 使用 Phoenix UI 创建的 traces/datasets 在容器重启后仍可访问

❌ **失败的标志：**
- 容器重启后 `/phoenix/workdir` 为空
- 主机挂载目录没有任何文件
- Phoenix UI 中的数据在重启后消失

## 🛠️ 故障排查

### 问题 1: 权限错误

```bash
# 错误信息
Permission denied: '/phoenix/workdir'

# 解决方案：确保挂载的主机目录有正确权限
chmod 777 /data/phoenix/workdir
```

### 问题 2: 数据未持久化

```bash
# 检查 volume 挂载
docker inspect phoenix | grep -A 10 Mounts

# 应该看到类似输出：
# "Mounts": [
#     {
#         "Type": "bind",
#         "Source": "/data/phoenix/workdir",
#         "Destination": "/phoenix/workdir",
#         ...
#     }
# ]
```

### 问题 3: 找不到数据库文件

```bash
# 检查工作目录
docker exec phoenix ls -la /phoenix/workdir

# 检查环境变量
docker exec phoenix env | grep PHOENIX
```

## 📝 环境变量参考

| 变量名 | 默认值 | 说明 |
|-------|--------|------|
| `PHOENIX_WORKING_DIR` | `/phoenix/workdir` | 工作目录路径（已在 Dockerfile 设置） |
| `PHOENIX_SQL_DATABASE_URL` | `sqlite://...` | 数据库连接字符串 |
| `PHOENIX_PORT` | `6006` | Web UI 端口 |
| `PHOENIX_GRPC_PORT` | `4317` | OTLP gRPC 端口 |

## 🔒 安全建议

1. **生产环境使用 PostgreSQL**: SQLite 适合开发和小规模部署
2. **备份数据**: 定期备份 `/phoenix/workdir` 或 PostgreSQL 数据库
3. **访问控制**: 使用 Phoenix 的认证功能限制访问
4. **网络隔离**: 不要将端口直接暴露到公网

## 📚 相关文档

- [Phoenix 官方部署文档](https://arize.com/docs/phoenix/self-hosting)
- [Dockerfile 源文件](../Dockerfile)
- [Docker Compose 配置示例](../docker-compose.yml)

## ❓ 常见问题

**Q: 我的平台只支持 `data:/phoenix/workdir` 格式，如何配置？**

A: Dockerfile 已经针对此格式优化，直接使用该挂载配置即可。环境变量 `PHOENIX_WORKING_DIR=/phoenix/workdir` 已内置。

**Q: 如何切换到 PostgreSQL？**

A: 添加环境变量 `PHOENIX_SQL_DATABASE_URL=postgresql://...`，同时仍需挂载 `/phoenix/workdir` 用于其他文件。

**Q: 容器使用 nonroot 用户，会有权限问题吗？**

A: 不会。Dockerfile 已将工作目录权限设置为 777，支持 nonroot 用户（UID 65532）。

**Q: 需要多大的存储空间？**

A: 取决于使用规模：
- 小规模（< 10K traces/天）: 10GB
- 中等规模（10K-100K traces/天）: 50GB
- 大规模（> 100K traces/天）: 建议使用 PostgreSQL + 100GB+

---

**最后更新**: 2025-10-31
**维护者**: Claude Code Analysis
