#!/bin/bash

# =============================================================================
# Python Project Cleanup Script
# 
# This script performs comprehensive cleanup of Python project artifacts
# including cache files, temporary files, and build artifacts.
# 
# Author: AI Assistant
# Date: $(date +%Y-%m-%d)
# =============================================================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Color codes for output formatting
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m' # No Color

# Logging functions
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

# Function to safely remove directories
safe_remove_dir() {
    local pattern="$1"
    local description="$2"
    
    if [[ -z "$pattern" ]]; then
        log_error "Pattern is empty for: $description"
        return 1
    fi
    
    log_info "Cleaning $description..."
    
    # Count items before removal
    local count=$(find . -type d -name "$pattern" 2>/dev/null | wc -l)
    
    if [[ $count -eq 0 ]]; then
        log_info "No $description found to clean"
        return 0
    fi
    
    # Remove with error handling
    if find . -type d -name "$pattern" -exec rm -rf {} + 2>/dev/null; then
        log_success "Removed $count $description"
    else
        log_warning "Some $description could not be removed (may be in use)"
    fi
}

# Function to safely remove files
safe_remove_files() {
    local pattern="$1"
    local description="$2"
    
    if [[ -z "$pattern" ]]; then
        log_error "Pattern is empty for: $description"
        return 1
    fi
    
    log_info "Cleaning $description..."
    
    # Count items before removal
    local count=$(find . -type f -name "$pattern" 2>/dev/null | wc -l)
    
    if [[ $count -eq 0 ]]; then
        log_info "No $description found to clean"
        return 0
    fi
    
    # Remove with error handling
    if find . -type f -name "$pattern" -delete 2>/dev/null; then
        log_success "Removed $count $description"
    else
        log_warning "Some $description could not be removed (may be in use)"
    fi
}

# Main cleanup function
main() {
    log_info "Starting Python project cleanup..."
    log_info "Working directory: $(pwd)"
    
    # Check if we're in a git repository
    if git rev-parse --git-dir > /dev/null 2>&1; then
        log_info "Git repository detected"
    else
        log_warning "Not in a git repository - be careful with cleanup"
    fi
    
    # Python cache directories
    safe_remove_dir "__pycache__" "Python cache directories"
    safe_remove_dir ".pytest_cache" "pytest cache directories"
    safe_remove_dir ".mypy_cache" "mypy cache directories"
    safe_remove_dir ".coverage" "coverage cache directories"
    
    # Python compiled files
    safe_remove_files "*.pyc" "Python compiled files"
    safe_remove_files "*.pyo" "Python optimized files"
    safe_remove_files "*.pyd" "Python extension modules"
    safe_remove_files "*.pkl" "Python pickle files"
    safe_remove_files "experiment_results_*.csv" "experiment_results_*.csv"
    safe_remove_files "analysis_experiment_results_*.json" "analysis_experiment_results_*.json"
    
    # Jupyter notebook checkpoints
    safe_remove_dir ".ipynb_checkpoints" "Jupyter notebook checkpoints"
    
    # Distribution and build artifacts
    safe_remove_dir "build" "build directories"
    safe_remove_dir "dist" "distribution directories"
    safe_remove_dir "experiment_visualizations" "experiment_visualizations"
    safe_remove_dir "*.egg-info" "egg-info directories"
    safe_remove_dir "__pycache__" "Python cache directories (recursive)"
    
    # Virtual environment artifacts (be careful)
    if [[ -d "venv" ]] || [[ -d ".venv" ]] || [[ -d "env" ]]; then
        log_warning "Virtual environment detected - skipping venv cleanup for safety"
    fi
    
    # IDE and editor files
    safe_remove_dir ".vscode" "VSCode settings"
    safe_remove_dir ".idea" "PyCharm/IntelliJ settings"
    safe_remove_files "*.swp" "Vim swap files"
    safe_remove_files "*.swo" "Vim swap files"
    safe_remove_files "*~" "Backup files"
    
    # Temporary files
    safe_remove_files "*.tmp" "Temporary files"
    safe_remove_files "*.temp" "Temporary files"
    safe_remove_files "experiment_results.csv" "experiment_results.csv"
    safe_remove_files ".DS_Store" "macOS system files"
    safe_remove_dir "runs" "runs"
    safe_remove_dir "test_runs" "test_runs"
    
    # Log files (optional - uncomment if needed)
    safe_remove_files "*.log" "Log files"
    
    log_success "Cleanup completed successfully!"
    
    # Optional: Show disk usage before and after
    if command -v du >/dev/null 2>&1; then
        log_info "Current directory size: $(du -sh . | cut -f1)"
    fi
}

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Python Project Cleanup Script

OPTIONS:
    -h, --help      Show this help message
    -v, --verbose   Enable verbose output
    -d, --dry-run   Show what would be cleaned without actually cleaning

EXAMPLES:
    $0              # Run normal cleanup
    $0 --dry-run    # Show what would be cleaned
    $0 --verbose    # Run with verbose output

This script cleans:
- Python cache directories (__pycache__, .pytest_cache, etc.)
- Compiled Python files (*.pyc, *.pyo)
- Build artifacts (build/, dist/, *.egg-info)
- IDE settings (.vscode, .idea)
- Temporary and backup files
- Jupyter notebook checkpoints

WARNING: This script will permanently delete files. Use --dry-run first.
EOF
}

# Parse command line arguments
VERBOSE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Set verbose mode
if [[ "$VERBOSE" == true ]]; then
    set -x
fi

# Check if this is a dry run
if [[ "$DRY_RUN" == true ]]; then
    log_warning "DRY RUN MODE - No files will be actually deleted"
    log_info "Would clean the following:"
    find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".mypy_cache" -o -name ".coverage" -o -name ".ipynb_checkpoints" -o -name "build" -o -name "dist" -o -name "*.egg-info" \) 2>/dev/null | head -20
    find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.pyd" -o -name "*.swp" -o -name "*.swo" -o -name "*~" -o -name "*.tmp" -o -name "*.temp" -o -name ".DS_Store" \) 2>/dev/null | head -20
    log_info "Use '$0' (without --dry-run) to actually perform cleanup"
    exit 0
fi

# Run main function
main "$@"