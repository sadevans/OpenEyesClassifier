from sklearn.metrics import roc_curve, auc
from utils import *
from OpenEyesClassificator import *


def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    frr = 1 - tpr
    abs_diffs = np.abs(fpr - frr)
    min_index = np.argmin(abs_diffs)
    eer = (fpr[min_index]+ frr[min_index])/2
    
    return eer


def compute_accuracy(preds, labels):
    correct = (preds == labels).sum().item()
    total = preds.shape[0]

    return correct/total


def plot_images(images, labels, scores, num_images=12):

    fig, axes = plt.subplots(num_images//4, num_images//3, figsize=(12,8))
    for i, ax in enumerate(axes.flat):
        if i<len(images):
            image = images[i]
            label = labels[i]
            prediction = scores[i]
            ax.imshow(image, cmap='gray')
            ax.set_title(f"Label: {label}, Prediction: {prediction:.4f}")
        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    return fig


def test_model(test_images, test_labels, plot=False, n_images=12, name=None):

    model = OpenEyesClassificator()
    test_acc = 0.0
    scores_list = []
    imgs_list = []

    for image_path, label in zip(test_images, test_labels):
        
        score = model.predict(image_path)
        scores_list.append(score)
        if plot:
            img = cv2.imread(image_path, 0)
            imgs_list.append(img)

    test_eer = compute_eer(test_labels, scores_list)
    preds = (torch.tensor(scores_list) > 0.5).float()
    test_acc = compute_accuracy(preds, torch.tensor(test_labels))

    if plot:
        if name is None:
            name='Testing_images'
        plot_images(imgs_list, test_labels, scores_list, num_images=n_images)
        plt.savefig(f'./figs/{name}.png')

    print(f"Testing accuracy = {test_acc:.4f}, testing eer = {test_eer:.4f}")


if __name__ == "__main__":

    open_dir = './dataset/hard_cases/open'
    close_dir = './dataset/hard_cases/close'
    images, labels = load_image_paths_and_labels(open_dir, close_dir)
    test_model(images, labels, plot=True, name='Hard cases test for final model')