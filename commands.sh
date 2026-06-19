# Start the cluster
k3d cluster start telcochurn
kubectl apply -f kubernetes/argocd/application.yaml

# Stop the cluster
k3d cluster stop telcochurn
