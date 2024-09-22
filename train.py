import torch
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import f1_score

    
def train_epoch(model, train_imgs, y_train, criterion, optimizer):
    """Train my model for num_epochs times"""
    model.train()

    y_color_train, y_cos_train = y_train
    color_criterion, cos_criterion = criterion

    # training
    num_epochs = 100
    for epoch in range(num_epochs):
        if torch.cuda.is_available():
            inputs = Variable(train_imgs).cuda()
            color_target = Variable(y_color_train).cuda()
            cos_target = Variable(y_cos_train).cuda()
        else:
            inputs = Variable(train_imgs)
            color_target = Variable(y_color_train)
            cos_target = Variable(y_cos_train)


        # forward
        color_out, cos_out = model(inputs)
        loss_color = color_criterion(color_out, color_target)
        loss_cos = cos_criterion(cos_out, cos_target)

        # total loss
        total_loss = loss_color + loss_cos

        # backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if (epoch+1) % 10 == 0:
            print(f'Epoch[{epoch+1}/{num_epochs}] Color Loss: {loss_color.item()}, Type Loss: {loss_cos.item()}, Total Loss: {total_loss.item()}')

def test_model(model, test_imgs, y_test):
    """Test my model (using F1-score)"""

    # Get my true labels(grand truth) for testing
    y_color_test, y_cos_test = y_test

    # Test my model after training
    model.eval()
    with torch.no_grad():
        outputs_color, outputs_cos = model(test_imgs)
        predicted_color = torch.sigmoid(outputs_color).cpu().numpy()
        predicted_cos = torch.sigmoid(outputs_cos).cpu().numpy()

        # Calculate F1-score bases on my prediction
        y_color_pred = np.argmax(predicted_color, axis=1)
        y_cos_pred = np.argmax(predicted_cos, axis=1)

        f1_color = f1_score(y_color_test, y_color_pred, average='weighted')
        f1_cos = f1_score(y_cos_test, y_cos_pred, average='weighted')

        print(f'F1-score for color classification: {f1_color}')
        print(f'F1-score for type classification: {f1_cos}')