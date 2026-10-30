import numpy as np
from utils import get_msg_mgr


def evaluate_scoliosis(data, dataset, metric='euc'):

    msg_mgr = get_msg_mgr()
    
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    logits = np.array(data['embeddings'])
    labels = data['types']
    
    # Label mapping: negative->0, neutral->1, positive->2  
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    true_ids = np.array([label_map[status] for status in labels])
    
    pred_ids = np.argmax(logits.mean(-1), axis=-1)
    
    # Calculate evaluation metrics
    # Total Accuracy: proportion of correctly predicted samples among all samples
    accuracy = accuracy_score(true_ids, pred_ids)
    
    # Macro-average Precision: average of precision scores for each class
    precision = precision_score(true_ids, pred_ids, average='macro', zero_division=0)
    
    # Macro-average Recall: average of recall scores for each class  
    recall = recall_score(true_ids, pred_ids, average='macro', zero_division=0)
    
    # Macro-average F1: average of F1 scores for each class
    f1 = f1_score(true_ids, pred_ids, average='macro', zero_division=0)
    
    # Print results
    msg_mgr.log_info(f"Total Accuracy: {accuracy*100:.2f}%")
    msg_mgr.log_info(f"Macro-avg Precision: {precision*100:.2f}%") 
    msg_mgr.log_info(f"Macro-avg Recall: {recall*100:.2f}%")
    msg_mgr.log_info(f"Macro-avg F1 Score: {f1*100:.2f}%")
    
    return {
        "scalar/test_accuracy/": accuracy,
        "scalar/test_precision/": precision, 
        "scalar/test_recall/": recall,
        "scalar/test_f1/": f1
    }
