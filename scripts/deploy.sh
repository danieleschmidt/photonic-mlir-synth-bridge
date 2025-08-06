#!/bin/bash
# Production deployment script for Photonic MLIR

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
NAMESPACE="photonic-mlir"
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-registry.company.com}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Validation functions
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi
    
    # Check docker (for building)
    if ! command -v docker &> /dev/null; then
        log_error "docker not found. Please install Docker."
        exit 1
    fi
    
    # Check kubectl connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Unable to connect to Kubernetes cluster."
        exit 1
    fi
    
    # Check helm (optional)
    if command -v helm &> /dev/null; then
        HELM_AVAILABLE=true
        log_info "Helm detected - advanced deployment options available"
    else
        HELM_AVAILABLE=false
        log_warning "Helm not found - using kubectl for deployment"
    fi
    
    log_success "Prerequisites check completed"
}

# Docker operations
build_image() {
    log_info "Building Docker image..."
    
    cd "$PROJECT_ROOT"
    
    # Build the image
    docker build -f docker/Dockerfile -t "photonic-mlir:${IMAGE_TAG}" .
    
    # Tag for registry if specified
    if [[ "$DOCKER_REGISTRY" != "registry.company.com" ]]; then
        docker tag "photonic-mlir:${IMAGE_TAG}" "${DOCKER_REGISTRY}/photonic-mlir:${IMAGE_TAG}"
        log_info "Tagged image for registry: ${DOCKER_REGISTRY}/photonic-mlir:${IMAGE_TAG}"
    fi
    
    log_success "Docker image built successfully"
}

push_image() {
    if [[ "$DOCKER_REGISTRY" == "registry.company.com" ]]; then
        log_warning "Using default registry - skipping push"
        return
    fi
    
    log_info "Pushing image to registry..."
    docker push "${DOCKER_REGISTRY}/photonic-mlir:${IMAGE_TAG}"
    log_success "Image pushed to registry"
}

# Kubernetes operations
create_namespace() {
    log_info "Creating namespace..."
    
    if kubectl get namespace "$NAMESPACE" &> /dev/null; then
        log_info "Namespace $NAMESPACE already exists"
    else
        kubectl apply -f "$PROJECT_ROOT/k8s/namespace.yaml"
        log_success "Namespace $NAMESPACE created"
    fi
}

deploy_secrets() {
    log_info "Deploying secrets..."
    
    # Check if secrets already exist
    if kubectl get secret photonic-mlir-secrets -n "$NAMESPACE" &> /dev/null; then
        log_warning "Secrets already exist - skipping creation"
        return
    fi
    
    # In production, you would generate secure secrets
    if [[ "$DEPLOYMENT_ENV" == "production" ]]; then
        log_warning "Production deployment detected"
        log_warning "Please ensure secrets are properly configured with secure values"
        log_warning "Current secrets are for development only"
    fi
    
    kubectl apply -f "$PROJECT_ROOT/k8s/secret.yaml"
    log_success "Secrets deployed"
}

deploy_config() {
    log_info "Deploying configuration..."
    kubectl apply -f "$PROJECT_ROOT/k8s/configmap.yaml"
    log_success "Configuration deployed"
}

deploy_storage() {
    log_info "Deploying storage..."
    kubectl apply -f "$PROJECT_ROOT/k8s/pvc.yaml"
    log_success "Storage deployed"
}

deploy_rbac() {
    log_info "Deploying RBAC..."
    kubectl apply -f "$PROJECT_ROOT/k8s/serviceaccount.yaml"
    log_success "RBAC deployed"
}

deploy_application() {
    log_info "Deploying application..."
    
    # Update image in deployment
    if [[ "$DOCKER_REGISTRY" != "registry.company.com" ]]; then
        # Replace image name in deployment file
        sed "s|photonic-mlir:latest|${DOCKER_REGISTRY}/photonic-mlir:${IMAGE_TAG}|g" \
            "$PROJECT_ROOT/k8s/deployment.yaml" | kubectl apply -f -
    else
        kubectl apply -f "$PROJECT_ROOT/k8s/deployment.yaml"
    fi
    
    kubectl apply -f "$PROJECT_ROOT/k8s/service.yaml"
    log_success "Application deployed"
}

deploy_ingress() {
    log_info "Deploying ingress..."
    
    # Check if ingress controller exists
    if ! kubectl get ingressclass nginx &> /dev/null; then
        log_warning "NGINX ingress controller not found - skipping ingress deployment"
        log_info "Install NGINX ingress controller: kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml"
        return
    fi
    
    kubectl apply -f "$PROJECT_ROOT/k8s/ingress.yaml"
    log_success "Ingress deployed"
}

deploy_autoscaling() {
    log_info "Deploying autoscaling..."
    
    # Check if metrics server is available
    if ! kubectl get deployment metrics-server -n kube-system &> /dev/null; then
        log_warning "Metrics server not found - HPA may not work properly"
        log_info "Install metrics server: kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml"
    fi
    
    kubectl apply -f "$PROJECT_ROOT/k8s/hpa.yaml"
    log_success "Autoscaling deployed"
}

# Wait for deployment
wait_for_deployment() {
    log_info "Waiting for deployment to be ready..."
    
    # Wait for deployment to be available
    kubectl rollout status deployment/photonic-mlir -n "$NAMESPACE" --timeout=300s
    
    # Wait for pods to be ready
    kubectl wait --for=condition=ready pod -l app=photonic-mlir -n "$NAMESPACE" --timeout=300s
    
    log_success "Deployment is ready"
}

# Health check
health_check() {
    log_info "Performing health check..."
    
    # Get service details
    local service_ip
    service_ip=$(kubectl get svc photonic-mlir -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    
    # Port forward for health check
    kubectl port-forward svc/photonic-mlir 8080:80 -n "$NAMESPACE" &
    local pf_pid=$!
    
    # Wait a moment for port forward to establish
    sleep 5
    
    # Perform health check
    if curl -f http://localhost:8080/health &> /dev/null; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        kill $pf_pid 2>/dev/null || true
        exit 1
    fi
    
    # Clean up port forward
    kill $pf_pid 2>/dev/null || true
}

# Show deployment status
show_status() {
    log_info "Deployment Status:"
    echo
    
    # Show pods
    echo "Pods:"
    kubectl get pods -n "$NAMESPACE" -l app=photonic-mlir
    echo
    
    # Show services
    echo "Services:"
    kubectl get svc -n "$NAMESPACE"
    echo
    
    # Show ingress if available
    if kubectl get ingress -n "$NAMESPACE" &> /dev/null; then
        echo "Ingress:"
        kubectl get ingress -n "$NAMESPACE"
        echo
    fi
    
    # Show HPA status
    if kubectl get hpa -n "$NAMESPACE" &> /dev/null; then
        echo "Horizontal Pod Autoscaler:"
        kubectl get hpa -n "$NAMESPACE"
        echo
    fi
    
    log_success "Photonic MLIR deployed successfully!"
    
    # Show connection information
    log_info "Connection Information:"
    if kubectl get ingress photonic-mlir-ingress -n "$NAMESPACE" &> /dev/null; then
        local ingress_host
        ingress_host=$(kubectl get ingress photonic-mlir-ingress -n "$NAMESPACE" -o jsonpath='{.spec.rules[0].host}')
        echo "  External URL: https://$ingress_host"
    fi
    echo "  Port Forward: kubectl port-forward svc/photonic-mlir 8080:80 -n $NAMESPACE"
    echo "  Logs: kubectl logs -f deployment/photonic-mlir -n $NAMESPACE"
}

# Rollback function
rollback() {
    log_warning "Rolling back deployment..."
    kubectl rollout undo deployment/photonic-mlir -n "$NAMESPACE"
    kubectl rollout status deployment/photonic-mlir -n "$NAMESPACE"
    log_success "Rollback completed"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up deployment..."
    
    read -p "Are you sure you want to delete the entire Photonic MLIR deployment? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kubectl delete namespace "$NAMESPACE"
        log_success "Cleanup completed"
    else
        log_info "Cleanup cancelled"
    fi
}

# Main deployment function
deploy() {
    log_info "Starting Photonic MLIR deployment to $DEPLOYMENT_ENV environment"
    
    check_prerequisites
    build_image
    push_image
    create_namespace
    deploy_secrets
    deploy_config
    deploy_storage
    deploy_rbac
    deploy_application
    deploy_ingress
    deploy_autoscaling
    wait_for_deployment
    health_check
    show_status
}

# Script usage
usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo
    echo "Commands:"
    echo "  deploy       Deploy Photonic MLIR (default)"
    echo "  rollback     Rollback to previous version"
    echo "  cleanup      Remove entire deployment"
    echo "  status       Show deployment status"
    echo "  logs         Show application logs"
    echo "  build        Build Docker image only"
    echo "  push         Push Docker image only"
    echo
    echo "Environment Variables:"
    echo "  DEPLOYMENT_ENV     Deployment environment (default: production)"
    echo "  DOCKER_REGISTRY    Docker registry URL (default: registry.company.com)"
    echo "  IMAGE_TAG          Docker image tag (default: latest)"
    echo
    echo "Examples:"
    echo "  $0 deploy"
    echo "  DEPLOYMENT_ENV=staging $0 deploy"
    echo "  IMAGE_TAG=v1.0.0 $0 deploy"
    echo "  DOCKER_REGISTRY=your-registry.com $0 deploy"
}

# Handle commands
case "${1:-deploy}" in
    deploy)
        deploy
        ;;
    rollback)
        rollback
        ;;
    cleanup)
        cleanup
        ;;
    status)
        show_status
        ;;
    logs)
        kubectl logs -f deployment/photonic-mlir -n "$NAMESPACE"
        ;;
    build)
        build_image
        ;;
    push)
        build_image
        push_image
        ;;
    -h|--help)
        usage
        ;;
    *)
        log_error "Unknown command: $1"
        usage
        exit 1
        ;;
esac