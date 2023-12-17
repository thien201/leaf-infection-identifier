# Imporiting Necessary Libraries
import tensorflow as tf
import numpy as np
from PIL import Image


# Cleanig image    
def clean_image(image):
    image = np.array(image)
    
    # Resizing the image
    image = np.array(Image.fromarray(
        image).resize((512, 512), Image.Resampling.LANCZOS))

        
    # Adding batch dimensions to the image
    # YOu are seeting :3, that's becuase sometimes user upload 4 channel image,
    image = image[np.newaxis, :, :, :3]
    # So we just take first  3 channels
    
    return image
    
    
def get_prediction(model, image):

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)
    
    # Inputting the image to keras generators
    test = datagen.flow(image)
    
    # Predict from the image
    predictions = model.predict(test)
    predictions_arr = np.array(np.argmax(predictions))
    
    return predictions, predictions_arr
    

# Making the final results 
def make_results(predictions, predictions_arr):
    
    result = {}
    if int(predictions_arr) == 0:
        result = {"status": " Lá cây khỏe, tốt không có bệnh ",
                    "prediction": f"{int(predictions[0][0].round(2)*100)}%"}
    if int(predictions_arr) == 1:
        result = {"status": ' Bị không chỉ đến một loại bệnh cụ thể mà là một thuật ngữ chung để mô tả tình trạng một cây bị nhiễm đồng thời bởi nhiều loại bệnh khác nhau. Các bệnh này có thể bao gồm sự kết hợp của bệnh do nấm, vi khuẩn, virus, hoặc thậm chí do ký sinh trùng hoặc thiếu hụt dinh dưỡng gây ra. ',
                    "prediction": f"{int(predictions[0][1].round(2)*100)}%"}
    if int(predictions_arr) == 2:
        result = {"status": ' bị bệnh gỉ sắt. Bệnh này thường xuất hiện dưới dạng các đốm màu cam đến nâu đỏ trên lá và thân cây, tương tự như màu của gỉ sắt, nó có tên tiếng Anh là "Rust". Nấm gây bệnh gỉ sắt thuộc chi Puccinia và một số chi khác, chúng phát triển và lan truyền trong điều kiện ẩm ướt ',
                    "prediction": f"{int(predictions[0][2].round(2)*100)}%"}
    if int(predictions_arr) == 3:
        result = {"status": ' bị bệnh do nấm hoặc vi khuẩn gây ra, và nó thường ảnh hưởng đến nhiều loại cây trồng khác nhau, bao gồm cả cây ăn trái như táo và lê, có tên tiếng Anh là Scab.Bệnh ghẻ thường xuất hiện dưới dạng các vết sần sùi, vảy, hoặc các đốm cứng trên bề mặt lá, quả, và cành non. Những vết này ban đầu có màu xanh lục đến nâu và sau đó có thể chuyển sang màu nâu đậm hơn hoặc đen ',
                    "prediction": f"{int(predictions[0][3].round(2)*100)}%"}
    return result   
