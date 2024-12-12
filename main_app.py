import streamlit as st
import app  # Underwater Waste Detection Model module
import seaborn as sns
import matplotlib.pyplot as plt
from inference import garbage  # Garbage classification results

# Define waste labels for the detection model
labels = ['Mask', 'can', 'cellphone', 'electronics', 'gbottle', 'glove', 
          'metal', 'net', 'pbag', 'pbottle', 'plastic', 
          'rod', 'sunglasses', 'tire']

def main():
    st.sidebar.title('Navigation')
    selected_model = st.sidebar.selectbox('', 
                                          ['Home', 'MARINE DEBRIS DETECTION ', 
                                           'Generated Report'])

    # Display content based on the selected model
    if selected_model == 'Home':
        st.title('Neural Ocean')
        st.image('./assets/yacht.jpg')
        st.success(
            'Neural Ocean addresses the issue of growing marine debris using YoloV8-based detection. '
            'The model was trained on a dataset of 3626 images, accurately identifying various underwater waste items.'
        )

    elif selected_model == 'MARINE DEBRIS DETECTION ':
        app.app()  # Run the waste detection model interface

    elif selected_model == 'Generated Report':
        st.header('Frequency of All Waste Labels')
        # Calculate occurrences for each label
        occurrences = [garbage.count(label) for label in labels]
        sns.barplot(y=labels, x=occurrences)
        plt.xlabel("Occurrences")
        plt.ylabel("Labels")
        plt.title("Histogram of Occurrences")
        st.pyplot()

        st.header("Conclusion:")
        most_frequent_label = labels[occurrences.index(max(occurrences))]
        st.success(
            f'The most frequently detected waste type is {most_frequent_label}, '
            f'appearing {max(occurrences)} times in recent observations.'
        )
    else:
        st.warning('Please select a model from the sidebar.')

# Run the Streamlit app
if __name__ == '__main__':
    main()
