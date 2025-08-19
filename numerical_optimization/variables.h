// -*- coding: utf-8 -*-
"""
@author: quantummole
"""


#ifndef VARIABLES_H
#define VARIABLES_H

namespace num_opt {
    #include <unordered_set>
    #include <functional>
    #include <memory>

    typedef double variable_type;

    class Variable: public std::enable_shared_from_this<Variable> {
    private:
        variable_type data;
        variable_type gradient;
        std::function<void()> _backward;
        std::unordered_set<std::shared_ptr<Variable>> parents;
        std::string op;

    public:
        Variable(variable_type data, std::unordered_set<std::shared_ptr<Variable>> parents = {}, std::string op = "");

        void set_grad(variable_type grad_value);
        float get_data();
        void set_data(variable_type data);
        float get_grad() const;
        std::unordered_set<std::shared_ptr<Variable>>  get_parents() const;

        std::shared_ptr<Variable> operator+(const std::shared_ptr<Value>& other);
        std::shared_ptr<Variable> operator-();
        std::shared_ptr<Variable> operator-(const std::shared_ptr<Value>& other);
        std::shared_ptr<Variable> pow(const std::shared_ptr<Value>& other);
        std::shared_ptr<Variable> operator/(const std::shared_ptr<Value>& other);
        std::shared_ptr<Variable> operator*(const std::shared_ptr<Value>& other);

        void backward();
    };

};

#endif 