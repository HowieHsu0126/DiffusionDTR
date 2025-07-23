#!/bin/bash

# 设置错误处理
set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查CUDA设备可用性
check_cuda_devices() {
    log_info "Checking CUDA device availability..."
    
    if ! command -v nvidia-smi &> /dev/null; then
        log_error "nvidia-smi not found. CUDA may not be available."
        return 1
    fi
    
    # 检查设备数量
    device_count=$(nvidia-smi --list-gpus | wc -l)
    log_info "Found $device_count CUDA devices"
    
    if [ "$device_count" -lt 2 ]; then
        log_warning "Only $device_count CUDA devices available, but we need at least 2 for parallel execution"
        log_warning "Some tasks may run on the same device"
    fi
    
    return 0
}

# 创建日志目录
log_info "Creating log directory..."
mkdir -p Output/logs

# 检查CUDA设备
check_cuda_devices

# 任务配置
# 当CUDA_VISIBLE_DEVICES=0时，只有GPU 0可见，所以DEVICE=cuda:0
# 当CUDA_VISIBLE_DEVICES=1时，只有GPU 1可见，所以DEVICE=cuda:0
declare -A task_configs=(
    ["iv"]="CUDA_VISIBLE_DEVICES=0 TASK=iv DEVICE=cuda:0"
    ["rrt"]="CUDA_VISIBLE_DEVICES=1 TASK=rrt DEVICE=cuda:0" 
    ["vent"]="CUDA_VISIBLE_DEVICES=1 TASK=vent DEVICE=cuda:0"
)

# 存储进程ID
declare -A pids

# 启动任务
for task in "${!task_configs[@]}"; do
    log_info "Starting $task task on ${task_configs[$task]}..."
    
    # 启动任务
    eval "${task_configs[$task]} nohup python -m Libs.exp.run_all > Output/logs/benchmark-$task.log 2>&1 &"
    pids[$task]=$!
    
    log_success "$task task started with PID: ${pids[$task]}"
done

# 显示启动信息
echo ""
log_success "All tasks started successfully!"
echo ""
echo "Task PIDs:"
for task in "${!pids[@]}"; do
    echo "  $task task: ${pids[$task]}"
done

echo ""
echo "Log files:"
for task in "${!pids[@]}"; do
    echo "  $task: Output/logs/benchmark-$task.log"
done

echo ""
echo "Monitoring commands:"
echo "  # Monitor all logs simultaneously:"
echo "  tail -f Output/logs/benchmark-*.log"
echo ""
echo "  # Monitor individual logs:"
for task in "${!pids[@]}"; do
    echo "  tail -f Output/logs/benchmark-$task.log"
done

echo ""
echo "Process management:"
echo "  # Check if processes are still running:"
echo "  ps -p ${pids[*]}"
echo ""
echo "  # Kill all tasks if needed:"
echo "  kill ${pids[*]}"

# 可选：等待所有任务完成
if [ "$1" = "--wait" ]; then
    echo ""
    log_info "Waiting for all tasks to complete..."
    for task in "${!pids[@]}"; do
        log_info "Waiting for $task task (PID: ${pids[$task]})..."
        wait ${pids[$task]}
        if [ $? -eq 0 ]; then
            log_success "$task task completed successfully"
        else
            log_error "$task task failed"
        fi
    done
    log_success "All tasks completed!"
fi 