import pandas as pd
import streamlit as st
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config(
    page_title="医疗费用预测",
    page_icon="🏥",
)

def introduce_page():
    """当选择简介页面时，将呈现该函数的内容"""
    st.write("# 欢迎使用")
    st.sidebar.success("单击预测医疗费用")
    st.markdown(
        """
        # 医疗费用预测应用
        
        这个应用利用机器学习模型来预测医疗费用，为保险公司的保险定价提供参考。
        
        ## 背景介绍
        - 开发目标: 帮助保险公司合理定价保险产品，控制风险
        - 模型算法: 利用随机森林回归算法训练医疗费用预测模型
        
        ## 使用指南
        - 输入准确完整的被保险人信息，可以得到更准确的费用预测
        - 预测结果可以作为保险定价的重要参考，但需审慎决策
        - 有任何问题欢迎联系我们的技术支持
        
        技术支持: 📧 support@example.com
        """
    )

def predict_page():
    """当选择预测费用页面时，将呈现该函数的内容"""
    st.markdown(
        """
        ## 使用说明
        这个应用利用机器学习模型来预测医疗费用，为保险公司的保险定价提供参考。
        - **输入信息**: 在下面输入被保险人的个人信息、疾病信息等
        - **费用预测**: 应用会预测被保险人的未来医疗费用支出
        """
    )
    
    # 运用表单和表单提交按钮
    with st.form('user_inputs'):
        age = st.number_input('年龄', min_value=0)
        sex = st.radio('性别', options=['男性', '女性'])
        bmi = st.number_input('BMI', min_value=0.0)
        children = st.number_input("子女数量：", step=1, min_value=0)
        smoke = st.radio("是否吸烟", ("是", "否"))
        region = st.selectbox('区域', ('东南部', '西南部', '东北部', '西北部'))
        submitted = st.form_submit_button('预测费用')
    
    if submitted:
        # 初始化数据预处理格式中与性别相关的变量
        sex_female, sex_male = 0, 0
        # 根据用户输入的性别数据更改对应的值
        if sex == '女性':
            sex_female = 1
        elif sex == '男性':
            sex_male = 1

        # 初始化数据预处理格式中与吸烟相关的变量
        smoke_yes, smoke_no = 0, 0
        # 根据用户输入的吸烟数据更改对应的值
        if smoke == '是':
            smoke_yes = 1
        elif smoke == '否':
            smoke_no = 1
        
        # 初始化数据预处理格式中与区域相关的变量
        region_northeast, region_southeast, region_northwest, region_southwest = 0, 0, 0, 0
        # 根据用户输入的区域数据更改对应的值
        if region == '东北部':
            region_northeast = 1
        elif region == '东南部':     
            region_southeast = 1
        elif region == '西北部':
            region_northwest = 1
        elif region == '西南部':
            region_southwest = 1

        # 整理特征数据
        format_data = [age, bmi, children, sex_female, sex_male,
                       smoke_no, smoke_yes,
                       region_northeast, region_southeast, region_northwest,
                       region_southwest]
        
        st.text(format_data)
        
        # 使用 pickle 的 load 方法从磁盘文件反序列化加载一个之前保存的随机森林回归模型
        try:
            with open('rfr_model.pkl', 'rb') as f:
                rfr_model = pickle.load(f)
            
            # 将特征数据转换为DataFrame
            format_data_df = pd.DataFrame(data=[format_data], columns=rfr_model.feature_names_in_)
            
            # 使用模型对格式化后的数据进行预测，返回预测的医疗费用
            predict_result = rfr_model.predict(format_data_df)[0]
            
            st.write('根据您输入的数据，预测该客户的医疗费用是：', round(predict_result, 2))
        except FileNotFoundError:
            st.error("模型文件未找到，请先训练模型。")
        except Exception as e:
            st.error(f"预测过程中出现错误：{str(e)}")


"""训练随机森林回归模型并保存"""
   
# 读取数据
insurance_df = pd.read_csv('insurance-chinese.csv', encoding='gbk')
insurance_df.info()
        
# 准备特征和目标变量
output = insurance_df['医疗费用']
features = insurance_df[['年龄', '性别', 'BMI', '子女数量', '是否吸烟', '区域']]
features = pd.get_dummies(features)
        
print(features.head())
print(output.head())
        
# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(features, output, train_size=0.8)
        
# 构建并训练随机森林回归模型
rfr = RandomForestRegressor()
rfr.fit(x_train, y_train)
        
# 评估模型
y_pred = rfr.predict(x_test)
r2 = r2_score(y_test, y_pred)
print(f'该模型的可决系数（R - squared）是：{r2}')
        
# 保存模型
with open('rfr_model.pkl', 'wb') as f:
    
    pickle.dump(rfr, f)
print('保存成功，已生成相关文件。')
        
# 在左侧添加侧边栏并设置单选按钮
nav = st.sidebar.radio("导航", ["简介", "预测医疗费用"])

# 根据选择的结果，展示不同的页面
if nav == "简介":
    introduce_page()
elif nav == "预测医疗费用":
    predict_page()
