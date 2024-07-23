{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "80ca32fd",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "    Outlook Temperature Humidity   Windy Play\n",
      "0     Sunny         Hot     High    Weak   No\n",
      "1     Sunny         Hot     High  Strong   No\n",
      "2  Overcast         Hot     High    Weak  Yes\n",
      "3      Rain        Mild     High    Weak  Yes\n",
      "4      Rain        Cool   Normal    Weak  Yes\n"
     ]
    },
    {
     "ename": "InvalidParameterError",
     "evalue": "The 'feature_names' parameter of plot_tree must be an instance of 'list' or None. Got array(['Outlook_Rain', 'Outlook_Sunny', 'Temperature_Hot',\n       'Temperature_Mild', 'Humidity_Normal', 'Windy_Weak'], dtype=object) instead.",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mInvalidParameterError\u001b[0m                     Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[3], line 31\u001b[0m\n\u001b[0;32m     29\u001b[0m \u001b[38;5;66;03m# Visualize the ID3 decision tree\u001b[39;00m\n\u001b[0;32m     30\u001b[0m plt\u001b[38;5;241m.\u001b[39mfigure(figsize\u001b[38;5;241m=\u001b[39m(\u001b[38;5;241m12\u001b[39m, \u001b[38;5;241m8\u001b[39m))\n\u001b[1;32m---> 31\u001b[0m plot_tree(clf_id3, filled\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mTrue\u001b[39;00m, feature_names\u001b[38;5;241m=\u001b[39mencoder\u001b[38;5;241m.\u001b[39mget_feature_names_out([\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mOutlook\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mTemperature\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mHumidity\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mWindy\u001b[39m\u001b[38;5;124m'\u001b[39m]), class_names\u001b[38;5;241m=\u001b[39m[\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mNo\u001b[39m\u001b[38;5;124m'\u001b[39m, \u001b[38;5;124m'\u001b[39m\u001b[38;5;124mYes\u001b[39m\u001b[38;5;124m'\u001b[39m])\n\u001b[0;32m     32\u001b[0m plt\u001b[38;5;241m.\u001b[39mshow()\n\u001b[0;32m     34\u001b[0m \u001b[38;5;66;03m# Predict the labels for the test set using ID3\u001b[39;00m\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\sklearn\\utils\\_param_validation.py:201\u001b[0m, in \u001b[0;36mvalidate_params.<locals>.decorator.<locals>.wrapper\u001b[1;34m(*args, **kwargs)\u001b[0m\n\u001b[0;32m    198\u001b[0m to_ignore \u001b[38;5;241m+\u001b[39m\u001b[38;5;241m=\u001b[39m [\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mself\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcls\u001b[39m\u001b[38;5;124m\"\u001b[39m]\n\u001b[0;32m    199\u001b[0m params \u001b[38;5;241m=\u001b[39m {k: v \u001b[38;5;28;01mfor\u001b[39;00m k, v \u001b[38;5;129;01min\u001b[39;00m params\u001b[38;5;241m.\u001b[39marguments\u001b[38;5;241m.\u001b[39mitems() \u001b[38;5;28;01mif\u001b[39;00m k \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;129;01min\u001b[39;00m to_ignore}\n\u001b[1;32m--> 201\u001b[0m validate_parameter_constraints(\n\u001b[0;32m    202\u001b[0m     parameter_constraints, params, caller_name\u001b[38;5;241m=\u001b[39mfunc\u001b[38;5;241m.\u001b[39m\u001b[38;5;18m__qualname__\u001b[39m\n\u001b[0;32m    203\u001b[0m )\n\u001b[0;32m    205\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[0;32m    206\u001b[0m     \u001b[38;5;28;01mwith\u001b[39;00m config_context(\n\u001b[0;32m    207\u001b[0m         skip_parameter_validation\u001b[38;5;241m=\u001b[39m(\n\u001b[0;32m    208\u001b[0m             prefer_skip_nested_validation \u001b[38;5;129;01mor\u001b[39;00m global_skip_validation\n\u001b[0;32m    209\u001b[0m         )\n\u001b[0;32m    210\u001b[0m     ):\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\sklearn\\utils\\_param_validation.py:95\u001b[0m, in \u001b[0;36mvalidate_parameter_constraints\u001b[1;34m(parameter_constraints, params, caller_name)\u001b[0m\n\u001b[0;32m     89\u001b[0m \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m     90\u001b[0m     constraints_str \u001b[38;5;241m=\u001b[39m (\n\u001b[0;32m     91\u001b[0m         \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;132;01m{\u001b[39;00m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m, \u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;241m.\u001b[39mjoin([\u001b[38;5;28mstr\u001b[39m(c)\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mfor\u001b[39;00m\u001b[38;5;250m \u001b[39mc\u001b[38;5;250m \u001b[39m\u001b[38;5;129;01min\u001b[39;00m\u001b[38;5;250m \u001b[39mconstraints[:\u001b[38;5;241m-\u001b[39m\u001b[38;5;241m1\u001b[39m]])\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m or\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m     92\u001b[0m         \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mconstraints[\u001b[38;5;241m-\u001b[39m\u001b[38;5;241m1\u001b[39m]\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m     93\u001b[0m     )\n\u001b[1;32m---> 95\u001b[0m \u001b[38;5;28;01mraise\u001b[39;00m InvalidParameterError(\n\u001b[0;32m     96\u001b[0m     \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mThe \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mparam_name\u001b[38;5;132;01m!r}\u001b[39;00m\u001b[38;5;124m parameter of \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mcaller_name\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m must be\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m     97\u001b[0m     \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mconstraints_str\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m. Got \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mparam_val\u001b[38;5;132;01m!r}\u001b[39;00m\u001b[38;5;124m instead.\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m     98\u001b[0m )\n",
      "\u001b[1;31mInvalidParameterError\u001b[0m: The 'feature_names' parameter of plot_tree must be an instance of 'list' or None. Got array(['Outlook_Rain', 'Outlook_Sunny', 'Temperature_Hot',\n       'Temperature_Mild', 'Humidity_Normal', 'Windy_Weak'], dtype=object) instead."
     ]
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 1200x800 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.model_selection import train_test_split, cross_val_score\n",
    "from sklearn.tree import DecisionTreeClassifier, plot_tree\n",
    "from sklearn.metrics import accuracy_score, classification_report\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "\n",
    "# Load the weather dataset from the local file\n",
    "# file_path = r\"/content/weather.csv\"\n",
    "df = pd.read_csv('weather.csv')\n",
    "\n",
    "# Display the first few rows of the dataset\n",
    "print(df.head())\n",
    "\n",
    "# Preprocessing: convert categorical variables to numerical using one-hot encoding\n",
    "encoder = OneHotEncoder(drop='first')\n",
    "X_encoded = encoder.fit_transform(df.drop('Play', axis=1)).toarray()\n",
    "y = df['Play']\n",
    "\n",
    "# Splitting the dataset into training and testing sets\n",
    "X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Initialize the decision tree classifier with ID3 algorithm\n",
    "clf_id3 = DecisionTreeClassifier(criterion='entropy', random_state=42)\n",
    "\n",
    "# Fit the ID3 classifier to the training data\n",
    "clf_id3.fit(X_train, y_train)\n",
    "\n",
    "# Visualize the ID3 decision tree\n",
    "plt.figure(figsize=(12, 8))\n",
    "plot_tree(clf_id3, filled=True, feature_names=encoder.get_feature_names_out(['Outlook', 'Temperature', 'Humidity', 'Windy']), class_names=['No', 'Yes'])\n",
    "plt.show()\n",
    "\n",
    "# Predict the labels for the test set using ID3\n",
    "y_pred_id3 = clf_id3.predict(X_test)\n",
    "\n",
    "# Evaluate the ID3 model\n",
    "accuracy_id3 = accuracy_score(y_test, y_pred_id3)\n",
    "report_id3 = classification_report(y_test, y_pred_id3)\n",
    "\n",
    "print(\"ID3 Algorithm Results:\")\n",
    "print(f\"Accuracy: {accuracy_id3}\")\n",
    "print(f\"Classification Report:\\n{report_id3}\")\n",
    "\n",
    "# Perform cross-validation for ID3\n",
    "cv_scores_id3 = cross_val_score(clf_id3, X_encoded, y, cv=5)\n",
    "print(\"Cross-Validation Scores (ID3):\", cv_scores_id3)\n",
    "print(\"Mean CV Accuracy (ID3):\", cv_scores_id3.mean())\n",
    "\n",
    "# Initialize the decision tree classifier with CART algorithm\n",
    "clf_cart = DecisionTreeClassifier(criterion='gini', random_state=42)\n",
    "\n",
    "# Fit the CART classifier to the training data\n",
    "clf_cart.fit(X_train, y_train)\n",
    "\n",
    "# Visualize the CART decision tree\n",
    "plt.figure(figsize=(12, 8))\n",
    "plot_tree(clf_cart, filled=True, feature_names=encoder.get_feature_names_out(['Outlook', 'Temperature', 'Humidity', 'Windy']).tolist(), class_names=['No', 'Yes'])\n",
    "plt.show()\n",
    "\n",
    "# Predict the labels for the test set using CART\n",
    "y_pred_cart = clf_cart.predict(X_test)\n",
    "\n",
    "# Evaluate the CART model\n",
    "accuracy_cart = accuracy_score(y_test, y_pred_cart)\n",
    "report_cart = classification_report(y_test, y_pred_cart)\n",
    "\n",
    "print(\"CART Algorithm Results:\")\n",
    "print(f\"Accuracy: {accuracy_cart}\")\n",
    "print(f\"Classification Report:\\n{report_cart}\")\n",
    "\n",
    "# Perform cross-validation for CART\n",
    "cv_scores_cart = cross_val_score(clf_cart, X_encoded, y, cv=5)\n",
    "print(\"Cross-Validation Scores (CART):\", cv_scores_cart)\n",
    "print(\"Mean CV Accuracy (CART):\", cv_scores_cart.mean())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1a6a831d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
